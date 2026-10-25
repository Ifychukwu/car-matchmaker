"""Deep links to UK car marketplaces, pre-filtered to the recommended model, years, budget and postcode.

No scraping: these are the same search pages a buyer would open by hand, so they always show today's stock and today's
prices from real dealers near the buyer. URL templates live in config/dealers.yaml so they can be fixed without code.
"""
from __future__ import annotations

from urllib.parse import urlencode

import yaml

from carmatch.data import ROOT

SITES = yaml.safe_load(open(ROOT / "config" / "dealers.yaml"))
GEARBOX = {"Automatic": "Automatic", "Semi-Auto": "Automatic", "Manual": "Manual"}


def build_links(make: str, model: str, *, postcode: str = "", radius: int = 50, price_max: float | None = None,
                year_from: int | None = None, year_to: int | None = None, max_mileage: int | None = None,
                fuel: str | None = None, transmission: str | None = None, sites=None) -> list[dict]:
    vals = {"make": make, "model": model, "postcode": postcode.replace(" ", "").upper(), "radius": radius,
            "price_max": int(price_max) if price_max else "", "year_from": year_from or "", "year_to": year_to or "",
            "max_mileage": int(max_mileage) if max_mileage else "", "fuel": fuel if fuel in ("Petrol", "Diesel", "Hybrid") else "",
            "transmission": GEARBOX.get(transmission or "", "")}
    if not vals["postcode"]:
        vals["radius"] = ""
    out = []
    for key, s in (sites or SITES).items():
        q = {}
        for k, tmpl in s["params"].items():
            v = str(tmpl).format(**vals).strip()
            if v and "{" not in v:
                q[k] = v
        out.append({"site": s["name"], "url": f"{s['base']}?{urlencode(q)}", "note": s.get("note", ""),
                    "filters": sorted(k for k in q if k not in ("_sacat", "_sop", "search_category", "include-delivery-option"))})
    return out
