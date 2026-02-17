# F1 Podium Probability ML

End-to-end machine learning system predicting Formula 1 race podium probability using time-aware modeling and strict leakage prevention.

## 🎯 Objective

Predict the probability that a given driver finishes on the podium (Top 3) for a race.

- Unit of prediction: (race, driver)
- Target: `is_podium = 1 if finish_position <= 3 else 0`
- Prediction time: Pre-weekend (before practice or qualifying)
- Train period: 2010–2024
- Test period: 2025

## 📦 Data Source

Data is sourced from the Ergast API schema via the Jolpica (Ergast-compatible) endpoint.

Raw race, results, driver, constructor, and circuit data are pulled and normalized into a structured modeling dataset.

## 🧠 Key ML Design Principles

- Time-aware feature engineering
- Strict data leakage prevention
- Rolling driver and constructor form features
- Proper probabilistic evaluation (Log Loss, PR-AUC, Calibration)

## 🏗 Project Structure

    data/
      raw/
      clean/
      modeling/

    src/
      ingest/
      clean/
      models/

    notebooks/
    reports/
    tests/


## ✅ Progress

- Raw data ingestion with pagination handling (Jolpica API)
- Clean normalized tables (drivers, constructors, circuits, races, results)
- Modeling table (driver-race level)
- Validated leakage-safe rolling features (driver + constructor form) in exploratory analysis
