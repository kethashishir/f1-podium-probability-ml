from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


RAW_DIR = Path("data/raw/ergast")
CLEAN_DIR = Path("data/clean")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_table(payload: Dict[str, Any], table_key: str) -> List[Dict[str, Any]]:
    """
    Ergast/Jolpica responses look like:
      payload["MRData"][...Table][table_key]
    where table_key can be:
      - "Races"
      - "Drivers"
      - "Constructors"
      - "Circuits"
    """
    mr = payload.get("MRData", {})

    # Try common tables in order. This avoids hardcoding too much.
    for table_name in ("RaceTable", "DriverTable", "ConstructorTable", "CircuitTable"):
        tbl = mr.get(table_name, {})
        if table_key in tbl:
            return tbl.get(table_key, [])

    # For results endpoints, races live in RaceTable -> Races, and each race has "Results"
    if table_key == "Races":
        rt = mr.get("RaceTable", {})
        return rt.get("Races", [])

    raise KeyError(f"Could not find table_key={table_key} in payload keys={list(mr.keys())}")


def build_dim_drivers() -> pd.DataFrame:
    payload = read_json(RAW_DIR / "drivers.json")
    drivers = _extract_table(payload, "Drivers")
    df = pd.json_normalize(drivers)

    # Standardize column names
    rename = {
        "driverId": "driver_id",
        "code": "driver_code",
        "givenName": "given_name",
        "familyName": "family_name",
        "dateOfBirth": "dob",
    }
    df = df.rename(columns=rename)

    keep = ["driver_id", "driver_code", "given_name", "family_name", "dob", "nationality", "url"]
    df = df[keep].copy()

    # Types
    df["driver_id"] = df["driver_id"].astype(str)
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # Uniqueness
    if df["driver_id"].duplicated().any():
        raise ValueError("Duplicate driver_id found in drivers")

    return df


def build_dim_constructors() -> pd.DataFrame:
    payload = read_json(RAW_DIR / "constructors.json")
    constructors = _extract_table(payload, "Constructors")
    df = pd.json_normalize(constructors)

    rename = {"constructorId": "constructor_id"}
    df = df.rename(columns=rename)

    keep = ["constructor_id", "name", "nationality", "url"]
    df = df[keep].copy()

    df["constructor_id"] = df["constructor_id"].astype(str)

    if df["constructor_id"].duplicated().any():
        raise ValueError("Duplicate constructor_id found in constructors")

    return df


def build_dim_circuits() -> pd.DataFrame:
    payload = read_json(RAW_DIR / "circuits.json")
    circuits = _extract_table(payload, "Circuits")
    df = pd.json_normalize(circuits)

    rename = {
        "circuitId": "circuit_id",
        "Location.locality": "locality",
        "Location.country": "country",
        "Location.lat": "lat",
        "Location.long": "lng",
    }
    df = df.rename(columns=rename)

    keep = ["circuit_id", "circuitName", "locality", "country", "lat", "lng", "url"]
    df = df[keep].copy()

    df["circuit_id"] = df["circuit_id"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")

    if df["circuit_id"].duplicated().any():
        raise ValueError("Duplicate circuit_id found in circuits")

    return df


def build_fct_races() -> pd.DataFrame:
    all_races: List[Dict[str, Any]] = []
    for year in range(2010, 2026):
        payload = read_json(RAW_DIR / f"races_{year}.json")
        races = _extract_table(payload, "Races")
        all_races.extend(races)

    df = pd.json_normalize(all_races)

    rename = {
        "raceName": "race_name",
        "Circuit.circuitId": "circuit_id",
        "Circuit.circuitName": "circuit_name",
        "date": "race_date",
    }
    df = df.rename(columns=rename)

    keep = ["season", "round", "race_name", "race_date", "circuit_id", "circuit_name", "url"]
    df = df[keep].copy()

    df["year"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.date
    df["circuit_id"] = df["circuit_id"].astype(str)

    # Create a stable race_id surrogate: year-round (works because round is unique within season)
    # In real warehouses you'd use provided raceId if available, but Ergast raceId isn't always directly present in races endpoint.
    df["race_id"] = df["year"].astype(str) + "_" + df["round"].astype(str)

    if df["race_id"].duplicated().any():
        raise ValueError("Duplicate race_id found in races")

    # Sanity
    if df["race_date"].isna().any():
        raise ValueError("Some races have missing race_date after parsing")

    return df[["race_id", "year", "round", "race_date", "race_name", "circuit_id", "circuit_name", "url"]]


def build_fct_results() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for year in range(2010, 2026):
        payload = read_json(RAW_DIR / f"results_{year}.json")
        races = _extract_table(payload, "Races")

        for race in races:
            season = race.get("season")
            rnd = race.get("round")
            race_id = f"{season}_{rnd}"

            results = race.get("Results", [])
            for r in results:
                flat = {
                    "race_id": race_id,
                    "year": int(season),
                    "round": int(rnd),
                    "driver_id": r.get("Driver", {}).get("driverId"),
                    "constructor_id": r.get("Constructor", {}).get("constructorId"),
                    "grid": r.get("grid"),
                    "position": r.get("position"),
                    "position_order": r.get("positionOrder"),
                    "points": r.get("points"),
                    "status": r.get("status"),
                    "laps": r.get("laps"),
                    "time": (r.get("Time") or {}).get("time"),
                }
                rows.append(flat)

    df = pd.DataFrame(rows)

    # Types
    df["race_id"] = df["race_id"].astype(str)
    df["driver_id"] = df["driver_id"].astype(str)
    df["constructor_id"] = df["constructor_id"].astype(str)

    for c in ["grid", "position", "position_order", "laps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    # Uniqueness
    if df.duplicated(subset=["race_id", "driver_id"]).any():
        dup = df[df.duplicated(subset=["race_id", "driver_id"], keep=False)].head(10)
        raise ValueError(f"Duplicate (race_id, driver_id) in results. Examples:\n{dup}")

    return df[
        [
            "race_id",
            "year",
            "round",
            "driver_id",
            "constructor_id",
            "grid",
            "position",
            "position_order",
            "points",
            "status",
            "laps",
            "time",
        ]
    ]


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    drivers = build_dim_drivers()
    constructors = build_dim_constructors()
    circuits = build_dim_circuits()
    races = build_fct_races()
    results = build_fct_results()

    drivers.to_parquet(CLEAN_DIR / "drivers.parquet", index=False)
    constructors.to_parquet(CLEAN_DIR / "constructors.parquet", index=False)
    circuits.to_parquet(CLEAN_DIR / "circuits.parquet", index=False)
    races.to_parquet(CLEAN_DIR / "races.parquet", index=False)
    results.to_parquet(CLEAN_DIR / "results.parquet", index=False)

    print("Wrote clean tables to data/clean/")
    print(" - drivers.parquet")
    print(" - constructors.parquet")
    print(" - circuits.parquet")
    print(" - races.parquet")
    print(" - results.parquet")


if __name__ == "__main__":
    main()
