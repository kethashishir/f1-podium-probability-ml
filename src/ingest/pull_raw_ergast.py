from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

from src.ingest.ergast_client import ErgastClient, ErgastConfig


RAW_DIR = Path("data/raw/ergast")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    client = ErgastClient(ErgastConfig())

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Dimension-style endpoints (single pulls)
    for name, endpoint in [
        ("drivers", "drivers.json"),
        ("constructors", "constructors.json"),
        ("circuits", "circuits.json"),
    ]:
        out_path = RAW_DIR / f"{name}.json"
        if out_path.exists():
            continue
        payload = client.fetch_raw(endpoint, params={"limit": 1000, "offset": 0})
        write_json(out_path, payload)

    # Fact endpoints per-year
    for year in tqdm(range(2010, 2026), desc="Pulling years"):
        races_path = RAW_DIR / f"races_{year}.json"
        results_path = RAW_DIR / f"results_{year}.json"

        if not races_path.exists():
            # races for a year: /{year}.json
            races = client.fetch_all(f"{year}.json")
            write_json(races_path, {"Races": races, "year": year})


        if not results_path.exists():
            # results for a year: /{year}/results.json
            races_with_results = client.fetch_all(f"{year}/results.json")
            write_json(results_path, {"Races": races_with_results, "year": year})



if __name__ == "__main__":
    main()
