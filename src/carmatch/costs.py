"""Running costs, clean-air-zone rules and total cost of ownership."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import yaml

from carmatch.data import ROOT

LITRES_PER_GALLON = 4.54609      # UK (imperial) gallon
CURRENT_YEAR = date.today().year


@dataclass
class Costs:
    fuel_price_per_litre: dict = field(default_factory=lambda: {"Petrol": 1.35, "Diesel": 1.42, "Hybrid": 1.35, "Other": 1.35})
    real_world_mpg_factor: float = 1.0
    market_index: float = 1.0
    ulez_min_year: dict = field(default_factory=lambda: {"Petrol": 2006, "Hybrid": 2006, "Diesel": 2016, "Other": 2016})

    @classmethod
    def load(cls, path=None) -> "Costs":
        p = path or ROOT / "config" / "costs.yaml"
        return cls(**yaml.safe_load(open(p))) if p.exists() else cls()


def annual_fuel_cost(mpg, fuel, miles: float, c: Costs) -> np.ndarray:
    mpg = np.asarray(pd.to_numeric(pd.Series(mpg), errors="coerce"), float) * c.real_world_mpg_factor
    ppl = np.array([c.fuel_price_per_litre.get(f, c.fuel_price_per_litre["Petrol"]) for f in pd.Series(fuel)], float)
    return miles / np.where(mpg > 0, mpg, np.nan) * LITRES_PER_GALLON * ppl


def today_year(age) -> np.ndarray:
    """A car that was `age` years old in the (2020) data corresponds to one registered in CURRENT_YEAR - age today."""
    return CURRENT_YEAR - np.asarray(age)


def ulez_ok(fuel, reg_year, c: Costs) -> np.ndarray:
    mins = np.array([c.ulez_min_year.get(f, 2016) for f in pd.Series(fuel)])
    return np.asarray(reg_year) >= mins


def ownership(df: pd.DataFrame, model, years: int, miles: float, c: Costs) -> pd.DataFrame:
    """Per listing: price today, fair range, yearly fuel and tax, resale after `years`, and cost per year of ownership."""
    fr = model.fair_range(df)
    price = df.price * c.market_index
    resale = model.value_at(df, years, miles) * c.market_index
    fuel = annual_fuel_cost(df.mpg, df.fuel, miles, c)
    tax = df.tax.fillna(df.tax.median()).to_numpy()
    out = pd.DataFrame({
        "price_now": price, "fair": fr.fair * c.market_index, "fair_low": fr.low * c.market_index, "fair_high": fr.high * c.market_index,
        "fuel_year": fuel, "tax_year": tax, "running_year": fuel + tax, "resale": resale,
        "depreciation": np.maximum(0, price - resale),
    }, index=df.index)
    out["depreciation_pct"] = out.depreciation / out.price_now
    out["cost_per_year"] = (out.depreciation + years * out.running_year) / years
    out["deal_ratio"] = out.price_now / out.fair            # < 1 = priced under the model's fair value
    return out
