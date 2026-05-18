import yaml
import json
import os


def generate_default_schedules(input_yaml: str, output_json: str):
    print(f"Reading topology from {input_yaml}...")

    with open(input_yaml, "r", encoding="utf-8") as f:
        district = yaml.safe_load(f)

    schedules = {}

    for building in district.get("buildings", []):
        b_id = building["id"]
        for apt in building.get("apartments", []):
            a_id = apt["id"]
            for room in apt.get("rooms", []):
                r_id = room["id"]
                room_key = f"{b_id}:{a_id}:{r_id}"

                temps_24h = [19] * 7 + [21] * 16 + [19]

                occ_24h = [1] * 8 + [0] * 8 + [1] * 8

                schedules[room_key] = {
                    "target_temp_c": temps_24h,
                    "max_co2_ppm": 1000,
                    "is_occupied": occ_24h,
                }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(schedules, f, indent=4)

    print(f"Success! Generated schedules for {len(schedules)} rooms in {output_json}")


if __name__ == "__main__":
    YAML_PATH = "config/district-definition.yml"
    JSON_PATH = "config/hvac-schedules.json"

    generate_default_schedules(YAML_PATH, JSON_PATH)
