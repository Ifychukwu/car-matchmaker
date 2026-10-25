"""Turn a buyer's needs into a ranked, explained shortlist of models, with dealer links.

1. Hard filters: budget, seats, body, size, fuel, gearbox, age, mileage, engine size, clean-air zone.
2. Every remaining listing gets a fair price, yearly fuel and tax, a resale estimate and a cost per year of ownership.
3. Listings are grouped by model and fuel; each group is scored on the buyer's priorities (percentile ranks within the
   shortlist, so scores are relative: "cheapest to run of what fits", not an absolute grade).
4. Top groups (one per model, at most two per brand) come back with reasons, trade-offs and pre-filtered links.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from carmatch.catalogue import AMBIGUOUS, CATALOGUE, SIZE_ORDER
from carmatch.costs import CURRENT_YEAR, Costs, ownership, ulez_ok
from carmatch.links import build_links
from carmatch.needs import PRIORITIES, Needs

ROOMY = {"estate": 0.6, "MPV": 1.0, "SUV": 0.6, "pickup": 0.4, "campervan": 1.0}


def with_attributes(df: pd.DataFrame) -> pd.DataFrame:
    att = pd.DataFrame([CATALOGUE.get(k, ("unknown", np.nan, "unknown")) for k in df.model_key], columns=["body", "seats", "size"], index=df.index)
    return pd.concat([df, att], axis=1)


def apply_filters(df: pd.DataFrame, n: Needs, c: Costs) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Returns matching listings and how many listings each requirement removed (in order), for 'why so few?' messages."""
    steps = []
    d = df
    price = d.price * c.market_index

    def keep(mask, label):
        nonlocal d
        mask = pd.Series(mask, index=d.index).fillna(False).astype(bool)
        steps.append((label, int((~mask).sum())))
        d = d[mask]

    keep((price <= n.budget) & (price >= n.budget_min), f"budget £{n.budget_min:,.0f}–£{n.budget:,.0f}")
    if n.seats:
        keep(d.seats >= n.seats, f"{n.seats}+ seats")
    if n.bodies:
        keep(d.body.isin(n.bodies), "body: " + "/".join(n.bodies))
    if n.sizes:
        keep(d["size"].isin(n.sizes), "size: " + "/".join(n.sizes))
    if n.fuels:
        keep(d.fuel.isin(n.fuels), "fuel: " + "/".join(n.fuels))
    if n.transmission == "Manual":
        keep(d.transmission == "Manual", "manual")
    elif n.transmission == "Automatic":
        keep(d.transmission.isin(["Automatic", "Semi-Auto"]), "automatic")
    if n.max_age is not None:
        keep(d.age <= n.max_age, f"≤ {n.max_age} years old")
    if n.max_mileage is not None:
        keep(d.mileage <= n.max_mileage, f"≤ {n.max_mileage:,} miles")
    if n.max_engine is not None:
        keep(d.engine_size.fillna(99) <= n.max_engine, f"engine ≤ {n.max_engine}L")
    if n.min_engine is not None:
        keep(d.engine_size.fillna(0) >= n.min_engine, f"engine ≥ {n.min_engine}L")
    if n.ulez:
        keep(ulez_ok(d.fuel, d.year, c), "clean-air-zone compliant")
    return d, steps


def _pct(s: pd.Series, higher_better=True) -> pd.Series:
    if len(s) <= 1:
        return pd.Series(1.0, index=s.index)
    r = s.rank(pct=True, method="average")
    return r if higher_better else 1 - r + 1 / len(s)


@dataclass
class Pick:
    model_key: str
    brand: str
    model: str
    fuel: str
    score: float
    stats: dict
    scores: dict
    reasons: list[str]
    tradeoffs: list[str]
    links: list[dict] = field(default_factory=list)
    examples: pd.DataFrame | None = None


def recommend(df: pd.DataFrame, n: Needs, price_model, c: Costs | None = None, k: int = 5, min_listings: int = 5,
              per_brand: int = 2) -> tuple[list[Pick], dict]:
    c = c or Costs()
    base = df if "body" in df else with_attributes(df)
    base = base[~base.model_key.isin(AMBIGUOUS)]
    cand, steps = apply_filters(base, n, c)
    info = {"matching_listings": len(cand), "filters": steps, "models_considered": 0}
    if cand.empty:
        return [], info
    own = ownership(cand, price_model, n.years_owned, n.annual_miles, c)
    cand = pd.concat([cand, own], axis=1)
    g = cand.groupby(["model_key", "fuel"])
    agg = g.agg(brand=("brand", "first"), model=("model", "first"), n=("price", "size"), price=("price_now", "median"),
                age=("age", "median"),                 mileage=("mileage", "median"), mileage_hi=("mileage", lambda m: np.quantile(m, 0.9)), mpg=("mpg", "median"),
                engine_l=("engine_size", "median"), seats=("seats", "first"), body=("body", "first"), size=("size", "first"),
                fuel_year=("fuel_year", "median"), tax_year=("tax_year", "median"), running=("running_year", "median"),
                dep_pct=("depreciation_pct", "median"), cost_year=("cost_per_year", "median"),
                good_deals=("deal_ratio", lambda r: float((r <= 0.97).mean())), deal_ratio=("deal_ratio", "median"),
                auto_share=("transmission", lambda t: float(t.isin(["Automatic", "Semi-Auto"]).mean())))
    agg = agg[agg.n >= min_listings]
    info["models_considered"] = int(agg.index.get_level_values(0).nunique())
    if agg.empty:
        return [], info
    space = agg["size"].map(SIZE_ORDER).fillna(1) + agg.seats.fillna(5) / 7 + agg.body.map(ROOMY).fillna(0)
    sc = pd.DataFrame({
        "running_costs": _pct(agg.running, False),
        "value": _pct(-agg.deal_ratio + agg.good_deals),
        "space": _pct(space),
        "performance": _pct(agg.engine_l.fillna(agg.engine_l.median())),
        "newer": (_pct(agg.age, False) + _pct(agg.mileage, False)) / 2,
        "availability": _pct(np.log1p(agg.n)),
        "low_depreciation": _pct(agg.dep_pct, False),
    })
    w = pd.Series({p: float(n.weights.get(p, 0)) for p in PRIORITIES})
    total = (sc[w.index] * w).sum(axis=1) / max(w.sum(), 1e-9)
    # a small, explicit nudge to the lowest cost of ownership so ties go to the cheaper car overall
    total = total + 0.05 * _pct(agg.cost_year, False)
    agg = agg.assign(score=total).sort_values("score", ascending=False)

    picks, seen, brands = [], set(), {}
    for (mk, fuel), row in agg.iterrows():
        if mk in seen or brands.get(row.brand, 0) >= per_brand:
            continue
        seen.add(mk)
        brands[row.brand] = brands.get(row.brand, 0) + 1
        s = sc.loc[(mk, fuel)]
        reasons, trade = _explain(row, s, w, n)
        # Links search today's market: model, budget, fuel, gearbox and postcode. Years are only set when the buyer
        # asked for a maximum age or a clean-air-zone car (the data's own years are 2020 asking prices, so they'd mislead).
        yf = None
        if n.ulez:
            yf = c.ulez_min_year.get(fuel, 2016)
        if n.max_age is not None:
            yf = max(yf or 0, CURRENT_YEAR - n.max_age)
        links = build_links(row.brand, row.model, postcode=n.postcode, radius=n.radius, price_max=n.budget, year_from=yf,
                            max_mileage=n.max_mileage, fuel=fuel, transmission=n.transmission if n.transmission != "any" else None)
        ex = cand[(cand.model_key == mk) & (cand.fuel == fuel)].sort_values("cost_per_year").head(5)
        picks.append(Pick(mk, row.brand, row.model, fuel, float(row.score), row.to_dict() | {"year_from": yf},
                          s.to_dict(), reasons, trade, links,
                          ex[["year", "age", "mileage", "transmission", "engine_size", "mpg", "price_now", "fair", "deal_ratio", "cost_per_year"]]))
        if len(picks) >= k:
            break
    return picks, info


def _explain(row, s: pd.Series, w: pd.Series, n: Needs) -> tuple[list[str], list[str]]:
    text = {
        "running_costs": f"About £{row.running:,.0f} a year on fuel and tax at {n.annual_miles:,} miles ({row.mpg:.0f} mpg listed)",
        "value": f"{row.good_deals:.0%} of matching adverts are priced under the model's fair value",
        "space": f"{int(row.seats) if pd.notna(row.seats) else '?'} seats, {row['size']} {row.body}",
        "performance": f"{row.engine_l:.1f}-litre engine (typical for the adverts that fit)",
        "newer": f"Typically {row.age:.0f} years old with {row.mileage:,.0f} miles",
        "availability": f"{int(row.n)} matching adverts in the data, so plenty to choose from",
        "low_depreciation": f"Expected to lose about {row.dep_pct:.0%} of its value over {n.years_owned} years",
    }
    weak = {
        "running_costs": f"Pricier to run than most of the shortlist (about £{row.running:,.0f} a year)",
        "value": "Adverts tend to be priced at or above fair value: negotiate",
        "space": f"Smaller than the alternatives ({row['size']} {row.body})",
        "performance": f"Modest engine ({row.engine_l:.1f}L)",
        "newer": f"Older or higher-mileage than the alternatives (typically {row.age:.0f} years, {row.mileage:,.0f} miles)",
        "availability": f"Rarer: only {int(row.n)} matching adverts, so expect to travel or wait",
        "low_depreciation": f"Loses value faster (about {row.dep_pct:.0%} over {n.years_owned} years)",
    }
    contrib = (s * w).sort_values(ascending=False)
    reasons = [text[p] for p in contrib.index if w[p] > 0 and s[p] >= 0.6][:3] or [text[contrib.index[0]]]
    trade = [weak[p] for p in (s * w).sort_values().index if w[p] >= 0.5 and s[p] <= 0.3][:2]
    return reasons, trade


RELAX = {"budget": ("budget", lambda n: setattr(n, "budget", n.budget * 1.25)), "seats": ("seats", lambda n: setattr(n, "seats", 0)),
         "body": ("bodies", lambda n: setattr(n, "bodies", [])), "size": ("sizes", lambda n: setattr(n, "sizes", [])),
         "fuel": ("fuels", lambda n: setattr(n, "fuels", [])), "gearbox": ("transmission", lambda n: setattr(n, "transmission", "any")),
         "age": ("max_age", lambda n: setattr(n, "max_age", None)), "mileage": ("max_mileage", lambda n: setattr(n, "max_mileage", None)),
         "engine": ("max_engine", lambda n: (setattr(n, "max_engine", None), setattr(n, "min_engine", None))),
         "clean-air zone": ("ulez", lambda n: setattr(n, "ulez", False))}


def relax_suggestions(df: pd.DataFrame, n: Needs, c: Costs | None = None, min_listings: int = 5) -> list[tuple[str, int]]:
    """For each requirement the buyer set, how many models would qualify if it were loosened (budget +25%, others dropped)."""
    import copy
    c = c or Costs()
    base = df if "body" in df else with_attributes(df)
    base = base[~base.model_key.isin(AMBIGUOUS)]
    default = Needs()
    out = []
    for label, (attr, fn) in RELAX.items():
        if label != "budget" and getattr(n, attr) == getattr(default, attr) and not (attr == "max_engine" and n.min_engine):
            continue
        m = copy.deepcopy(n)
        fn(m)
        d, _ = apply_filters(base, m, c)
        models = int((d.groupby(["model_key", "fuel"]).size() >= min_listings).sum()) if len(d) else 0
        out.append((("raise the budget to £%s" % f"{m.budget:,.0f}") if label == "budget" else f"drop the {label} requirement", models))
    return sorted(out, key=lambda x: -x[1])


def why_empty(info: dict, suggestions: list[tuple[str, int]] | None = None) -> str:
    """Plain-English explanation of what to loosen when nothing fits."""
    k = info["matching_listings"]
    msg = ("No adverts in the data meet every requirement." if k == 0 else
           f"Only {k} adverts in the data meet every requirement, too few to recommend from.")
    good = [s for s in (suggestions or []) if s[1] > 0]
    if good:
        msg += " Options: " + "; ".join(f"{a} ({k} models)" for a, k in good[:3]) + "."
    return msg
