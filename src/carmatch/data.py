"""Load the Kaggle "100,000 UK Used Car Data set" (one CSV per brand) into one clean table.

Source: https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes
Files used: audi, bmw, ford, hyundi, merc, skoda, toyota, vauxhall, vw (.csv), with columns
    model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize
(hyundi.csv calls the tax column "tax(£)"). cclass.csv and focus.csv duplicate rows of merc/ford with fewer columns,
and the "unclean" files are raw scrapes, so they are skipped.
"""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
BRANDS = {"audi": "Audi", "bmw": "BMW", "ford": "Ford", "hyundi": "Hyundai", "merc": "Mercedes-Benz", "skoda": "Skoda",
          "toyota": "Toyota", "vauxhall": "Vauxhall", "vw": "Volkswagen"}
COLUMNS = ["brand", "model", "year", "price", "transmission", "mileage", "fuel", "tax", "mpg", "engine_size"]


def _read_brand(fh, brand_key: str) -> pd.DataFrame:
    df = pd.read_csv(fh)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"tax(£)": "tax", "fuelType": "fuel", "engineSize": "engine_size"})
    df["brand"] = BRANDS[brand_key]
    return df


def load_raw(raw: Path = RAW) -> pd.DataFrame:
    frames = []
    for z in raw.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                key = Path(name).stem.lower()
                if key in BRANDS:
                    frames.append(_read_brand(BytesIO(zf.read(name)), key))
    if not frames:
        for p in raw.glob("*.csv"):
            if p.stem.lower() in BRANDS:
                frames.append(_read_brand(p, p.stem.lower()))
    if not frames:
        raise FileNotFoundError(f"no brand CSVs found in {raw}: download the Kaggle zip (see data/README.md)")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, as_of_year: int = 2020) -> tuple[pd.DataFrame, dict]:
    """Returns the cleaned table and a log of what was removed (and why)."""
    log = {"raw_rows": int(len(df))}
    d = df.copy()
    d["model"] = d["model"].astype(str).str.strip()
    for c in ("year", "price", "mileage", "tax", "mpg", "engine_size"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["fuel"] = d["fuel"].astype(str).str.strip()
    d["transmission"] = d["transmission"].astype(str).str.strip()
    before = len(d)
    d = d.drop_duplicates(subset=COLUMNS[:-1] + ["engine_size"])
    log["duplicates_removed"] = before - len(d)
    bad = (d.year > as_of_year) | (d.year < 1990) | (d.price <= 500) | (d.mileage < 0) | (d.mpg > 250) | d[["year", "price", "mileage"]].isna().any(axis=1)
    log["implausible_removed"] = int(bad.sum())        # e.g. a 2060 registration year, £0 prices, 470 mpg typos
    d = d[~bad]
    # The scrape has 3 "Electric" rows (two 2.0-litre Mondeos, probably hybrids, and an Ampera range-extender) and 9 "Other" gearboxes:
    # too few to model, so they are dropped. The data effectively has no EVs; the app says so rather than guessing.
    odd = (d.fuel == "Electric") | (d.transmission == "Other")
    log["too_rare_removed"] = int(odd.sum())
    d = d[~odd]
    # engine size 0 is a missing value (there are no electric cars left after the step above)
    d["engine_size"] = d.engine_size.where(d.engine_size > 0).astype(float)
    d["age"] = as_of_year - d["year"]
    d["model_key"] = d["brand"] + " " + d["model"]
    log["rows"] = int(len(d))
    log["models"] = int(d.model_key.nunique())
    return d.reset_index(drop=True)[COLUMNS + ["age", "model_key"]], log


def load(raw: Path = RAW, as_of_year: int = 2020) -> tuple[pd.DataFrame, dict]:
    return clean(load_raw(raw), as_of_year)
