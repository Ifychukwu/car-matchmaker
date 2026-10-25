"""Small synthetic data in the exact Kaggle schema, plus API payloads in the documented formats."""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

SPEC = {  # file key: [(model, base price new, engine, mpg, fuel options)]
    "ford": [(" Fiesta", 15000, 1.0, 60, ["Petrol", "Diesel"]), (" Focus", 20000, 1.5, 55, ["Petrol", "Diesel"]),
             (" S-MAX", 30000, 2.0, 48, ["Diesel"]), (" Ka+", 11000, 1.2, 57, ["Petrol"])],
    "vw": [(" Golf", 22000, 1.5, 55, ["Petrol", "Diesel"]), (" Up", 11000, 1.0, 64, ["Petrol"]), (" Touran", 26000, 1.6, 52, ["Diesel"])],
    "toyota": [(" Yaris", 16000, 1.5, 70, ["Hybrid", "Petrol"]), (" RAV4", 32000, 2.5, 55, ["Hybrid"])],
    "hyundi": [(" Tucson", 26000, 1.7, 52, ["Diesel", "Petrol"])],
}


def listings(n_per_model: int = 120, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {}
    for key, models in SPEC.items():
        rows = []
        for name, new, eng, mpg, fuels in models:
            for _ in range(n_per_model):
                age = int(rng.integers(0, 12))
                miles = int(max(10, age * rng.normal(9000, 2500) + rng.normal(2000, 800)))
                fuel = fuels[int(rng.integers(len(fuels)))]
                trans = rng.choice(["Manual", "Automatic", "Semi-Auto"], p=[0.6, 0.25, 0.15])
                price = new * 0.85 ** age * (1 - miles / 400000) * (1.06 if trans != "Manual" else 1) * (1.04 if fuel == "Hybrid" else 1)
                price *= np.exp(rng.normal(0, 0.06))
                rows.append({"model": name, "year": 2020 - age, "price": int(price), "transmission": trans, "mileage": miles,
                             "fuelType": fuel, "tax": 145 if age < 3 else 30, "mpg": round(mpg + rng.normal(0, 2), 1), "engineSize": eng})
        df = pd.DataFrame(rows)
        if key == "hyundi":
            df = df.rename(columns={"tax": "tax(£)"})
        out[key] = df
    # rows the cleaner must drop
    out["ford"] = pd.concat([out["ford"], pd.DataFrame([
        {"model": " Focus", "year": 2060, "price": 15000, "transmission": "Manual", "mileage": 100, "fuelType": "Petrol", "tax": 145, "mpg": 55, "engineSize": 1.0},
        {"model": " Fiesta", "year": 2015, "price": 0, "transmission": "Manual", "mileage": 30000, "fuelType": "Petrol", "tax": 30, "mpg": 60, "engineSize": 1.0},
        {"model": " Fiesta", "year": 2016, "price": 7000, "transmission": "Other", "mileage": 30000, "fuelType": "Petrol", "tax": 30, "mpg": 60, "engineSize": 1.0},
    ])], ignore_index=True)
    return out


def write_zip(path, frames: dict[str, pd.DataFrame]):
    with zipfile.ZipFile(path, "w") as z:
        for k, df in frames.items():
            z.writestr(f"{k}.csv", df.to_csv(index=False))
        z.writestr("unclean focus.csv", "model,year\n Focus,2016\n")    # must be ignored
    return path


def ebay_payload():
    return {"total": 3, "itemSummaries": [
        {"title": "2017 Ford Focus 1.5 TDCi Titanium 5dr, 42,000 miles, FSH", "price": {"value": "7995.00", "currency": "GBP"},
         "itemWebUrl": "https://www.ebay.co.uk/itm/111", "itemLocation": {"city": "Leeds", "postalCode": "LS6"}, "buyingOptions": ["FIXED_PRICE"],
         "image": {"imageUrl": "https://i.ebayimg.com/1.jpg"}, "seller": {"username": "dealer1"}},
        {"title": "Ford Focus 67 reg 1.0 EcoBoost 18k miles", "price": {"value": "9500.00", "currency": "GBP"},
         "itemWebUrl": "https://www.ebay.co.uk/itm/222", "itemLocation": {"city": "York"}, "buyingOptions": ["AUCTION"]},
        {"title": "Ford Focus spares or repair", "currentBidPrice": {"value": "450.00", "currency": "GBP"},
         "itemWebUrl": "https://www.ebay.co.uk/itm/333", "buyingOptions": ["AUCTION"]},
    ]}


def dvla_payload(**kw):
    d = {"registrationNumber": "AB12CDE", "taxStatus": "Taxed", "taxDueDate": "2027-03-01", "motStatus": "Valid", "make": "FORD",
         "yearOfManufacture": 2015, "engineCapacity": 1499, "co2Emissions": 105, "fuelType": "DIESEL", "markedForExport": False,
         "colour": "BLUE", "euroStatus": "EURO 5", "monthOfFirstRegistration": "2015-04"}
    d.update(kw)
    return d


def mot_payload():
    def t(day, res, miles, defects=(), expiry=""):
        return {"completedDate": f"{day}T10:00:00.000Z", "testResult": res, "odometerValue": str(miles), "odometerUnit": "MI",
                "odometerResultType": "READ", "expiryDate": expiry, "defects": list(defects)}
    adv = {"text": "Nearside Front Tyre worn close to legal limit (5.2.3 (e))", "type": "ADVISORY", "dangerous": False}
    return {"registration": "AB12CDE", "make": "FORD", "model": "FOCUS", "motTests": [
        t("2023-04-02", "PASSED", 61000, [adv], "2024-04-01"),
        t("2018-04-03", "PASSED", 30500),
        t("2019-04-01", "FAILED", 41000, [{"text": "Brake pipe corroded", "type": "MAJOR", "dangerous": False}]),
        t("2019-04-08", "PASSED", 41010),
        t("2021-04-05", "PASSED", 52000, [adv]),
        t("2022-04-01", "PASSED", 49000, [{"text": "Brake disc worn", "type": "DANGEROUS", "dangerous": True}]),
    ]}


class FakeResponse:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _hit(self, method, url, **kw):
        self.calls.append((method, url, kw))
        for frag, resp in self.routes.items():
            if frag in url:
                return resp
        return FakeResponse({}, 404)

    def get(self, url, **kw):
        return self._hit("GET", url, **kw)

    def post(self, url, **kw):
        return self._hit("POST", url, **kw)
