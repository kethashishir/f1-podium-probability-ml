from __future__ import annotations

from pathlib import Path

from duckdb import df
import pandas as pd


CLEAN_DIR = Path("data/clean")
MODEL_DIR = Path("data/modeling")


def compute_is_dnf(status: pd.Series) -> pd.Series:
    """
    Conservative DNF heuristic:
    - Treat "Finished" as not DNF
    - Treat "+1 Lap", "+2 Laps", etc. as not DNF
    - Everything else => DNF

    We'll validate/adjust this after inspecting status values in EDA.
    """
    s = status.fillna("").astype(str).str.strip()
    finished = s.eq("Finished")
    lapped_finish = s.str.match(r"^\+\d+\s+Lap(s)?$", na=False)
    return (~(finished | lapped_finish)).astype(int)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    races = pd.read_parquet(CLEAN_DIR / "races.parquet")
    results = pd.read_parquet(CLEAN_DIR / "results.parquet")

    # Join results to races to get race_date and circuit_id (source of truth)
    df = results.merge(
        races[["race_id", "race_date", "circuit_id", "race_name"]],
        on="race_id",
        how="left",
        validate="many_to_one",
    )

    # Basic sanity
    if df["race_date"].isna().any():
        missing = df[df["race_date"].isna()][["race_id", "year", "round"]].drop_duplicates().head(10)
        raise ValueError(f"Missing race_date after join. Examples:\n{missing}")

    # Labels / targets
    df["finish_position"] = df["finish_position"].astype("Int64")
    df["is_podium"] = (df["finish_position"].fillna(999) <= 3).astype(int)

    df["is_dnf"] = compute_is_dnf(df["status"])

    # Split
    df["split"] = "train"
    df.loc[df["year"] == 2025, "split"] = "test"

    # Keep columns (v1 modeling base table)
    out = df[
        [
            "race_id",
            "year",
            "round",
            "race_date",
            "race_name",
            "circuit_id",
            "driver_id",
            "constructor_id",
            "finish_position",
            "points",
            "status",
            "is_dnf",
            "is_podium",
            "split",
            # NOTE: grid exists in results but we will NOT use it as a feature in v1 (pre-weekend).
            # Keep it out of modeling table to reduce accidental leakage.
        ]
    ].copy()

    # Sort for time-aware processing later
    out = out.sort_values(["race_date", "race_id", "driver_id"]).reset_index(drop=True)

    # Uniqueness check
    if out.duplicated(subset=["race_id", "driver_id"]).any():
        dup = out[out.duplicated(subset=["race_id", "driver_id"], keep=False)].head(10)
        raise ValueError(f"Duplicate (race_id, driver_id) in modeling table. Examples:\n{dup}")

    # Write
    out_path = MODEL_DIR / "driver_race_modeling.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Wrote modeling table: {out_path}")
    print(f"Rows: {len(out):,}  Columns: {out.shape[1]}")
    print(out["split"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
