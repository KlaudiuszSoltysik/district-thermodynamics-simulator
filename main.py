import asyncio
import logging

import psycopg2
import tomllib
from psycopg2.extras import Json, execute_values

from modules.Simulator import SimulationStep, Simulator


def setup_logger():
    logger = logging.getLogger("dts")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("simulation_logs.txt", mode="w")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


def get_db_credentials(filepath):
    with open(filepath, "rb") as f:
        config = tomllib.load(f)
        return config["timescaledb"]


def insert_batch(conn, query, data):
    with conn.cursor() as cursor:
        execute_values(cursor, query, data)
    conn.commit()


def reset_database(db_config, logger):
    try:
        logger.info("Resetting TimescaleDB database...")
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS simulation_telemetry CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS mpc_forecast CASCADE;")

        cursor.execute("""
            CREATE TABLE simulation_telemetry (
                time TIMESTAMPTZ NOT NULL,
                out_temperature_c DOUBLE PRECISION,
                out_wind_speed_m_s DOUBLE PRECISION,
                out_wind_direction_deg DOUBLE PRECISION,
                out_sun_radiation_w_m2 DOUBLE PRECISION,
                out_sun_altitude_deg DOUBLE PRECISION,
                out_sun_azimuth_deg DOUBLE PRECISION,
                out_co2_ppm INTEGER,
                sys_electricity_price DOUBLE PRECISION,
                sys_gas_price DOUBLE PRECISION,
                sys_pv_yield_kw DOUBLE PRECISION,
                sys_cop_heating DOUBLE PRECISION,
                sys_cop_cooling DOUBLE PRECISION,
                room_temperatures_c JSONB,
                room_co2_ppm JSONB,
                room_q_hvac_w JSONB,
                room_q_hvac_perc JSONB,
                room_v_hvac_m3_s JSONB,
                meter_readings JSONB
            );
        """)
        cursor.execute("SELECT create_hypertable('simulation_telemetry', 'time');")

        cursor.execute("""
            CREATE TABLE mpc_forecast (
                time TIMESTAMPTZ PRIMARY KEY,
                planned_q_w JSONB,
                planned_v_m3_s JSONB,
                planned_t_target_c JSONB,
                planned_co2_max_ppm JSONB,
                planned_is_occupied JSONB
            );
        """)

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database reset successfully.")
    except Exception as e:
        logger.error(f"Failed to reset database: {e}.")
        raise


async def db_writer_worker(queue, db_config, batch_size, logger):
    telemetry_batch = []

    conn = psycopg2.connect(**db_config)
    logger.info("Database background writer started.")

    insert_telemetry_query = """
        INSERT INTO simulation_telemetry (
            time, out_temperature_c, out_wind_speed_m_s, out_wind_direction_deg, 
            out_sun_radiation_w_m2, out_sun_altitude_deg, out_sun_azimuth_deg, out_co2_ppm,
            sys_electricity_price, sys_gas_price, sys_pv_yield_kw, sys_cop_heating, sys_cop_cooling,
            room_temperatures_c, room_co2_ppm, room_q_hvac_w, room_q_hvac_perc, room_v_hvac_m3_s, meter_readings
        ) VALUES %s
    """

    upsert_forecast_query = """
        INSERT INTO mpc_forecast (time, planned_q_w, planned_v_m3_s, planned_t_target_c, planned_co2_max_ppm, planned_is_occupied)
        VALUES %s
        ON CONFLICT (time) DO UPDATE SET
            planned_q_w = EXCLUDED.planned_q_w,
            planned_v_m3_s = EXCLUDED.planned_v_m3_s,
            planned_t_target_c = EXCLUDED.planned_t_target_c,
            planned_co2_max_ppm = EXCLUDED.planned_co2_max_ppm,
            planned_is_occupied = EXCLUDED.planned_is_occupied;
    """

    try:
        while True:
            item = await queue.get()

            if item is None:
                break

            if not isinstance(item, SimulationStep):
                logger.warning("Received invalid item in queue, skipping.")
                queue.task_done()
                continue

            telemetry_batch.append(
                (
                    item.time,
                    item.out_temperature_c,
                    item.out_wind_speed_m_s,
                    item.out_wind_direction_deg,
                    item.out_sun_radiation_w_m2,
                    item.out_sun_altitude_deg,
                    item.out_sun_azimuth_deg,
                    item.out_co2_ppm,
                    item.sys_electricity_price,
                    item.sys_gas_price,
                    item.sys_pv_yield_kw,
                    item.sys_cop_heating,
                    item.sys_cop_cooling,
                    Json(item.room_temperatures_c),
                    Json(item.room_co2_ppm),
                    Json(item.room_q_hvac_w),
                    Json(item.room_q_hvac_perc),
                    Json(item.room_v_hvac_m3_s),
                    Json(item.meter_readings),
                )
            )

            if item.mpc_forecast:
                current_horizon_data = [
                    (
                        f["time"],
                        Json(f["q_w"]),
                        Json(f["v_m3_s"]),
                        Json(f["t_target_c"]),
                        Json(f["co2_max_ppm"]),
                        Json(f["is_occupied"]),
                    )
                    for f in item.mpc_forecast
                ]
                logger.info(
                    f"[{item.time}] MPC forecast saved (Horizon: {len(item.mpc_forecast)} steps)"
                )
                await asyncio.to_thread(
                    insert_batch, conn, upsert_forecast_query, current_horizon_data
                )

            if len(telemetry_batch) >= batch_size:
                await asyncio.to_thread(
                    insert_batch, conn, insert_telemetry_query, telemetry_batch
                )
                logger.info(
                    f"[{item.time}] Batch of {batch_size} telemetry steps saved."
                )
                telemetry_batch.clear()

            queue.task_done()

        if telemetry_batch:
            await asyncio.to_thread(
                insert_batch, conn, insert_telemetry_query, telemetry_batch
            )

    except Exception as e:
        logger.error(f"Error in database writer worker: {e}")
    finally:
        conn.close()
        logger.info("Database background writer shut down cleanly.")


async def main(
    credentials_path,
    district_config_path,
    weather_path,
    prices_path,
    hvac_schedule_patch,
    dt_seconds=300,
    batch_size=12,
):
    logger = setup_logger()
    logger.info("Starting simulation script...")

    db_config = get_db_credentials(credentials_path)

    reset_database(db_config, logger)

    data_queue = asyncio.Queue()

    writer_task = asyncio.create_task(
        db_writer_worker(data_queue, db_config, batch_size, logger=logger)
    )

    sim = Simulator(
        district_config_path=district_config_path,
        weather_path=weather_path,
        prices_path=prices_path,
        hvac_schedule_patch=hvac_schedule_patch,
        logger=logger,
    )

    logger.info("Entering main physics loop...")

    while True:
        result = sim.run_step(dt_seconds)

        if not result:
            logger.info("Physics loop finished.")
            break

        await data_queue.put(result)

        await asyncio.sleep(0)

    await data_queue.put(None)

    await writer_task

    logger.info("Simulation script completed successfully.")


if __name__ == "__main__":
    asyncio.run(
        main(
            "config/credentials.toml",
            "config/district-definition.yml",
            "config/weather-history.csv",
            "config/prices-history.csv",
            "config/hvac-schedules.json",
            dt_seconds=300,
            batch_size=12,
        )
    )
