"""Fair-price model for used cars, with honest baselines and a calibrated "fair range".

    python -m carmatch.price_model      # trains, evaluates on a frozen test split, saves models/price_model.pkl + reports/metrics.json

Model: gradient-boosted trees on log(price) from brand, model, age, mileage, fuel, gearbox, engine size, mpg and tax.
Baselines a buyer could do by hand:
  * model median:            the median asking price of that model
  * model + year median:     the median price of that model and registration year (falls back to the model median)
Fair range: split-conformal interval on the log residuals from a calibration split, so "80% range" means that ~80% of
held-out listings fall inside it (checked on the test set, not assumed).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from carmatch.data import ROOT

MODELS, REPORTS = ROOT / "models", ROOT / "reports"
CATS = ["brand", "model_key", "fuel", "transmission"]
NUMS = ["age", "mileage", "engine_size", "mpg", "tax"]
FEATURES = CATS + NUMS
SEED = 42
DEAL_BANDS = [  # (label, test on price / fair price, colour)
    ("Great price", "below the fair range", "#1baf7a"),
    ("Good price", "3%+ under fair value", "#6cc49a"),
    ("Fair price", "within 3% of fair value", "#8a8984"),
    ("A bit high", "3%+ over fair value", "#eb9a34"),
    ("Overpriced", "above the fair range", "#d64b3a"),
]


def _frame(df: pd.DataFrame, cats: dict[str, list] | None = None) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for c in CATS:
        v = x[c].astype(str)
        known = (cats or {}).get(c)
        x[c] = pd.Categorical(v.where(v.isin(known)) if known is not None else v, categories=known)   # unseen -> missing
    for c in NUMS:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)
    return x


@dataclass
class PriceModel:
    gbm: HistGradientBoostingRegressor | None = None
    cats: dict[str, list] = field(default_factory=dict)
    q80: float = 0.15          # conformal half-width in log space
    as_of_year: int = 2020

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame | None = None, **params) -> "PriceModel":
        self.cats = {c: sorted(train[c].astype(str).unique()) for c in CATS}
        p = dict(max_iter=800, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=10, l2_regularization=0.1,
                 categorical_features="from_dtype", random_state=SEED)
        p.update(params)
        self.gbm = HistGradientBoostingRegressor(**p).fit(_frame(train, self.cats), np.log(train.price))
        if calib is not None and len(calib):
            r = np.abs(np.log(calib.price.to_numpy()) - self._log_pred(calib))
            n = len(r)
            self.q80 = float(np.quantile(r, min(1.0, np.ceil(0.8 * (n + 1)) / n)))
        return self

    def _log_pred(self, df: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict(_frame(df, self.cats))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self._log_pred(df))

    def fair_range(self, df: pd.DataFrame) -> pd.DataFrame:
        lp = self._log_pred(df)
        return pd.DataFrame({"fair": np.exp(lp), "low": np.exp(lp - self.q80), "high": np.exp(lp + self.q80)}, index=df.index)

    def value_at(self, df: pd.DataFrame, years_later: float, annual_miles: float) -> np.ndarray:
        """Predicted price after `years_later` more years and `annual_miles` per year: the resale estimate."""
        f = df.copy()
        f["age"] = f["age"] + years_later
        f["mileage"] = f["mileage"] + years_later * annual_miles
        return self.predict(f)

    def save(self, path=None):
        path = path or MODELS / "price_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path=None) -> "PriceModel":
        with open(path or MODELS / "price_model.pkl", "rb") as fh:
            return pickle.load(fh)


def deal_label(price: float, fair: float, low: float, high: float) -> str:
    if price < low:
        return "Great price"
    if price > high:
        return "Overpriced"
    r = price / fair
    return "Good price" if r <= 0.97 else "A bit high" if r >= 1.03 else "Fair price"


# ---------------------------------------------------------------- baselines
class MedianBaseline:
    """Median price by group, falling back through coarser groups (then the overall median)."""

    def __init__(self, levels: list[list[str]]):
        self.levels = levels

    def fit(self, df: pd.DataFrame) -> "MedianBaseline":
        self.tables = [df.groupby(lv).price.median() for lv in self.levels]
        self.overall = float(df.price.median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = pd.Series(np.nan, index=df.index)
        for lv, tab in zip(self.levels, self.tables):
            keys = pd.MultiIndex.from_frame(df[lv]) if len(lv) > 1 else df[lv[0]]
            vals = pd.Series(tab.reindex(keys).to_numpy(), index=df.index)
            out = out.fillna(vals)
        return out.fillna(self.overall).to_numpy()


BASELINES = {
    "model_median": (["model_key"], ["brand"]),
    "model_year_median": (["model_key", "year"], ["model_key"], ["brand"]),
}
NAMES = {"gbm": "Gradient boosting (all features)", "model_year_median": "Median for model + year",
         "model_median": "Median for model"}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ape = np.abs(p - y) / y
    return {"mae": float(np.mean(np.abs(p - y))), "mape": float(np.mean(ape) * 100), "median_ape": float(np.median(ape) * 100),
            "within_10pct": float(np.mean(ape <= 0.10) * 100), "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}


def split(df: pd.DataFrame):
    """Frozen split: 70% train, 10% calibration (for the fair range), 20% test (scored once)."""
    rest, test = train_test_split(df, test_size=0.2, random_state=SEED)
    train, calib = train_test_split(rest, test_size=0.125, random_state=SEED)
    return train, calib, test


def evaluate(df: pd.DataFrame) -> tuple[PriceModel, dict, pd.DataFrame]:
    train, calib, test = split(df)
    model = PriceModel().fit(train, calib)
    preds = {"gbm": model.predict(test)}
    for k, lv in BASELINES.items():
        preds[k] = MedianBaseline([list(x) for x in lv]).fit(train).predict(test)
    y = test.price.to_numpy()
    out = {"overall": {k: metrics(y, p) for k, p in preds.items()}}
    fr = model.fair_range(test)
    out["fair_range"] = {"target_coverage": 80, "test_coverage": float(((y >= fr.low) & (y <= fr.high)).mean() * 100),
                         "half_width_pct": float((np.exp(model.q80) - 1) * 100)}
    t = test.assign(**{f"pred_{k}": v for k, v in preds.items()})
    out["by_brand"] = {b: {k: metrics(g.price.to_numpy(), g[f"pred_{k}"].to_numpy())["mape"] for k in preds}
                       for b, g in t.groupby("brand")}
    bands = pd.cut(t.price, [0, 7500, 12500, 20000, 35000, 1e9], labels=["<£7.5k", "£7.5–12.5k", "£12.5–20k", "£20–35k", "£35k+"])
    out["by_price_band"] = {str(b): {"n": int(len(g)), **{k: metrics(g.price.to_numpy(), g[f"pred_{k}"].to_numpy())["mape"] for k in preds}}
                            for b, g in t.groupby(bands, observed=True)}
    out["sizes"] = {"train": len(train), "calibration": len(calib), "test": len(test)}
    return model, out, t


def main():
    from carmatch.data import load
    df, log = load()
    model, out, t = evaluate(df)
    out["data"] = log
    REPORTS.mkdir(exist_ok=True)
    t[["brand", "model_key", "year", "mileage", "fuel", "transmission", "price"] + [c for c in t if c.startswith("pred_")]] \
        .to_csv(REPORTS / "test_predictions.csv.gz", index=False)
    # final model: refit on train + test as well, keeping the calibration split for the fair range
    train, calib, test = split(df)
    final = PriceModel().fit(pd.concat([train, test]), calib)
    final.save()
    out["final_model"] = {"rows": len(train) + len(test), "fair_range_half_width_pct": float((np.exp(final.q80) - 1) * 100)}
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    o = out["overall"]
    for k in o:
        print(f"{NAMES[k]:36s} MAE £{o[k]['mae']:,.0f}  MAPE {o[k]['mape']:.1f}%  within 10% {o[k]['within_10pct']:.0f}%  R² {o[k]['r2']:.3f}")
    print("fair range:", out["fair_range"])


if __name__ == "__main__":
    # run through the real module name so the saved model unpickles as carmatch.price_model.PriceModel, not __main__
    from carmatch.price_model import main as _main
    _main()
