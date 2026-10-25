"""Optional live data, all through official APIs with the user's own free keys (nothing is scraped).

* eBay Browse API: today's car listings for a recommended model, each run through the fair-price model.
    EBAY_CLIENT_ID / EBAY_CLIENT_SECRET   (developer.ebay.com, "Production" keyset)
* DVLA Vehicle Enquiry Service: tax, MOT status, Euro standard, CO2 and first registration for a number plate.
    DVLA_API_KEY                          (register at the DVLA developer portal)
* DVSA MOT History API: every MOT test with mileage, failures and advisories.
    MOT_CLIENT_ID / MOT_CLIENT_SECRET / MOT_API_KEY / MOT_TENANT_ID   (apply at documentation.history.mot.api.gov.uk)

Each client takes an injectable `session` (anything with .get/.post like requests), so tests run offline on fixtures.
"""
from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from datetime import date

import requests

UA = {"User-Agent": "car-matchmaker/1.0 (portfolio project)"}
TIMEOUT = 20


# ---------------------------------------------------------------- eBay
class EbayClient:
    TOKEN = "https://api.ebay.com/identity/v1/oauth2/token"
    SEARCH = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    CARS_CATEGORY = "9801"

    def __init__(self, client_id=None, client_secret=None, session=None):
        self.id = client_id or os.environ.get("EBAY_CLIENT_ID", "")
        self.secret = client_secret or os.environ.get("EBAY_CLIENT_SECRET", "")
        self.s = session or requests.Session()
        self._token = None

    @property
    def configured(self) -> bool:
        return bool(self.id and self.secret)

    def token(self) -> str:
        if not self._token:
            auth = base64.b64encode(f"{self.id}:{self.secret}".encode()).decode()
            r = self.s.post(self.TOKEN, headers={"Authorization": f"Basic {auth}", "Content-Type": "application/x-www-form-urlencoded", **UA},
                            data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}, timeout=TIMEOUT)
            r.raise_for_status()
            self._token = r.json()["access_token"]
        return self._token

    def search(self, make: str, model: str, price_max: float | None = None, limit: int = 50) -> list[dict]:
        filt = ["itemLocationCountry:GB", "buyingOptions:{FIXED_PRICE|AUCTION}"]
        if price_max:
            filt.append(f"price:[..{int(price_max)}],priceCurrency:GBP")
        r = self.s.get(self.SEARCH, headers={"Authorization": f"Bearer {self.token()}", "X-EBAY-C-MARKETPLACE-ID": "EBAY_GB", **UA},
                       params={"q": f"{make} {model}", "category_ids": self.CARS_CATEGORY, "filter": ",".join(filt), "limit": limit},
                       timeout=TIMEOUT)
        r.raise_for_status()
        return [parse_ebay_item(i) for i in r.json().get("itemSummaries", [])]


YEAR_RE = re.compile(r"\b(19[9]\d|20[0-3]\d)\b")
REG_PLATE_RE = re.compile(r"\b(\d{2}) ?(?:reg|plate)\b", re.I)
MILES_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{1,3}(?:\.\d)?k|\d{4,6})\s?(?:miles|mls|mi)\b", re.I)
ENGINE_RE = re.compile(r"\b([0-6]\.\d)\s?(?:l\b|litre|tdi|tsi|tfsi|ecoboost|d\b|i\b|hdi|cdti|crdi|t\b)?", re.I)


def plate_year(two: str) -> int | None:
    """UK age identifiers: '17' = March 2017, '67' = September 2017."""
    n = int(two)
    if 2 <= n <= 49:
        return 2000 + n
    if 51 <= n <= 99:
        return 2000 + n - 50
    return None


def parse_ebay_item(i: dict) -> dict:
    title = i.get("title", "")
    year = None
    if m := YEAR_RE.search(title):
        year = int(m.group(1))
    elif m := REG_PLATE_RE.search(title):
        year = plate_year(m.group(1))
    miles = None
    if m := MILES_RE.search(title):
        v = m.group(1).lower().replace(",", "")
        miles = int(float(v[:-1]) * 1000) if v.endswith("k") else int(v)
    eng = None
    if m := ENGINE_RE.search(title):
        eng = float(m.group(1))
    price = i.get("price") or i.get("currentBidPrice") or {}
    loc = i.get("itemLocation", {})
    return {"title": title, "price": float(price.get("value", "nan")), "url": i.get("itemWebUrl", ""),
            "image": (i.get("image") or {}).get("imageUrl", ""), "location": ", ".join(x for x in [loc.get("city"), loc.get("postalCode")] if x),
            "auction": "AUCTION" in (i.get("buyingOptions") or []), "year": year, "mileage": miles, "engine_size": eng,
            "seller": (i.get("seller") or {}).get("username", "")}


def price_ebay(items: list[dict], brand: str, model_name: str, fuel: str, transmission: str, price_model, ref) -> list[dict]:
    """Add a fair price and deal label to eBay items whose title gives the year and mileage.

    Other features (mpg, tax, engine) come from the median of the same model in the reference data. Fair prices are
    in 2020 money: compare the *ratio*, not the pound figure, unless the market index has been set."""
    import pandas as pd
    from carmatch.price_model import deal_label
    sub = ref[(ref.brand == brand) & (ref.model == model_name)]
    if sub.empty:
        return items
    med = sub[["mpg", "tax", "engine_size"]].median()
    rows = [it for it in items if it["year"] and it["mileage"] is not None and it["price"] == it["price"]]
    if not rows:
        return items
    as_of = price_model.as_of_year
    f = pd.DataFrame({"brand": brand, "model_key": f"{brand} {model_name}", "fuel": fuel, "transmission": transmission,
                      "age": [max(0, as_of - r["year"]) for r in rows], "mileage": [r["mileage"] for r in rows],
                      "engine_size": [r["engine_size"] or med.engine_size for r in rows], "mpg": med.mpg, "tax": med.tax})
    fr = price_model.fair_range(f)
    for r, (_, x) in zip(rows, fr.iterrows()):
        r.update(fair=x.fair, deal=deal_label(r["price"], x.fair, x.low, x.high))
    return items


# ---------------------------------------------------------------- DVLA + MOT
class DvlaClient:
    URL = "https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles"

    def __init__(self, api_key=None, session=None):
        self.key = api_key or os.environ.get("DVLA_API_KEY", "")
        self.s = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def lookup(self, reg: str) -> dict:
        r = self.s.post(self.URL, headers={"x-api-key": self.key, "Content-Type": "application/json", **UA},
                        json={"registrationNumber": clean_reg(reg)}, timeout=TIMEOUT)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()


class MotClient:
    BASE = "https://history.mot.api.gov.uk/v1/trade/vehicles/registration/"
    SCOPE = "https://tapi.dvsa.gov.uk/.default"

    def __init__(self, client_id=None, client_secret=None, api_key=None, tenant_id=None, session=None):
        e = os.environ
        self.id, self.secret = client_id or e.get("MOT_CLIENT_ID", ""), client_secret or e.get("MOT_CLIENT_SECRET", "")
        self.key, self.tenant = api_key or e.get("MOT_API_KEY", ""), tenant_id or e.get("MOT_TENANT_ID", "")
        self.s = session or requests.Session()
        self._token = None

    @property
    def configured(self) -> bool:
        return all([self.id, self.secret, self.key, self.tenant])

    def token(self) -> str:
        if not self._token:
            r = self.s.post(f"https://login.microsoftonline.com/{self.tenant}/oauth2/v2.0/token",
                            data={"grant_type": "client_credentials", "client_id": self.id, "client_secret": self.secret, "scope": self.SCOPE},
                            timeout=TIMEOUT)
            r.raise_for_status()
            self._token = r.json()["access_token"]
        return self._token

    def history(self, reg: str) -> dict:
        r = self.s.get(self.BASE + clean_reg(reg), headers={"Authorization": f"Bearer {self.token()}", "X-API-Key": self.key, **UA},
                       timeout=TIMEOUT)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()


def clean_reg(reg: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", reg.upper())


@dataclass
class MotSummary:
    tests: int = 0
    fails: int = 0
    advisories: int = 0
    dangerous: int = 0
    last_mileage: int | None = None
    miles_per_year: float | None = None
    mileage_drops: list[str] = field(default_factory=list)   # possible clocking or recording errors
    last_result: str = ""
    expiry: str = ""
    recurring: list[str] = field(default_factory=list)       # advisories repeated across tests
    timeline: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _miles(t: dict) -> int | None:
    try:
        v = int(str(t.get("odometerValue", "")).replace(",", ""))
    except ValueError:
        return None
    return int(v * 0.621371) if str(t.get("odometerUnit", "MI")).upper() == "KM" else v


def summarise_mot(h: dict) -> MotSummary:
    """Mileage history, failures, advisories and red flags from a DVSA MOT history response."""
    s = MotSummary()
    tests = sorted(h.get("motTests") or [], key=lambda t: t.get("completedDate", ""))
    s.tests = len(tests)
    if not tests:
        s.flags.append("No MOT tests on record (under 3 years old, or a new import)")
        return s
    seen: dict[str, int] = {}
    prev = None
    for t in tests:
        defects = t.get("defects") or []
        adv = [d for d in defects if str(d.get("type", "")).upper() in ("ADVISORY", "MINOR")]
        s.advisories += len(adv)
        s.dangerous += sum(1 for d in defects if d.get("dangerous") or str(d.get("type", "")).upper() == "DANGEROUS")
        failed = str(t.get("testResult", "")).upper().startswith("FAIL")
        s.fails += failed
        mi = _miles(t)
        day = str(t.get("completedDate", ""))[:10]
        if mi is not None and prev is not None and mi < prev[1]:
            s.mileage_drops.append(f"{day}: {mi:,} miles after {prev[1]:,} on {prev[0]}")
        if mi is not None:
            prev = (day, mi)
        for d in adv:
            key = re.sub(r"\s*\(.*", "", str(d.get("text", ""))).strip().lower()[:60]
            seen[key] = seen.get(key, 0) + 1
        s.timeline.append({"date": day, "result": "FAIL" if failed else "PASS", "mileage": mi, "advisories": len(adv),
                           "defects": len(defects) - len(adv)})
    last = tests[-1]
    s.last_result = "FAIL" if str(last.get("testResult", "")).upper().startswith("FAIL") else "PASS"
    s.expiry = str(last.get("expiryDate", "") or "")
    s.last_mileage = prev[1] if prev else None
    dated = [(r["date"], r["mileage"]) for r in s.timeline if r["mileage"] is not None and r["date"]]
    if len(dated) >= 2:
        d0, d1 = date.fromisoformat(dated[0][0]), date.fromisoformat(dated[-1][0])
        yrs = (d1 - d0).days / 365.25
        if yrs > 0.5:
            s.miles_per_year = (dated[-1][1] - dated[0][1]) / yrs
    s.recurring = [k for k, v in sorted(seen.items(), key=lambda x: -x[1]) if v >= 2][:5]
    if s.mileage_drops:
        s.flags.append("Mileage went DOWN between tests: possible clocking or a recording error, ask for proof")
    if s.dangerous:
        s.flags.append(f"{s.dangerous} dangerous defect(s) recorded in its history: check they were fixed")
    if s.last_result == "FAIL":
        s.flags.append("Failed its most recent MOT")
    if s.recurring:
        s.flags.append("Same advisory on more than one test (" + "; ".join(s.recurring[:2]) + "): likely never fixed")
    if s.miles_per_year and s.miles_per_year > 20000:
        s.flags.append(f"High use: about {s.miles_per_year:,.0f} miles a year")
    return s


def dvla_flags(v: dict) -> list[str]:
    """Red flags and a clean-air-zone (ULEZ) check from a DVLA record."""
    if not v:
        return ["Registration not found by the DVLA"]
    out = []
    if v.get("taxStatus") in ("Untaxed", "SORN"):
        out.append(f"Tax status: {v['taxStatus']}")
    if v.get("motStatus") == "Not valid":
        out.append("No valid MOT")
    if v.get("markedForExport"):
        out.append("Marked for export")
    fuel = str(v.get("fuelType", "")).upper()
    if fuel == "ELECTRICITY":
        return out + ["ULEZ: meets the standard (fully electric)"]
    need = 6 if "DIESEL" in fuel else 4
    euro = re.search(r"(\d)", str(v.get("euroStatus") or ""))
    if euro:
        num = int(euro.group(1))
        out.append(("ULEZ: meets" if num >= need else "ULEZ: does NOT meet") + f" the standard (Euro {num}; {fuel.title()} needs Euro {need}+)")
    elif v.get("yearOfManufacture"):
        ok = v["yearOfManufacture"] >= (2016 if need == 6 else 2006)
        out.append(f"ULEZ: {'probably meets' if ok else 'probably does NOT meet'} the standard (no Euro status on record, judged by year; confirm with TfL)")
    return out
