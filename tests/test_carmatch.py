import json
import types

import numpy as np
import pytest

import fixtures as fx
from carmatch import data, links, live
from carmatch.costs import Costs, annual_fuel_cost, ulez_ok
from carmatch.needs import Needs, parse_lifestyle, parse_with_llm, valid_postcode
from carmatch.price_model import MedianBaseline, PriceModel, deal_label, metrics, split
from carmatch.recommend import apply_filters, recommend, relax_suggestions, why_empty, with_attributes


@pytest.fixture(scope="module")
def cars(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    fx.write_zip(d / "cars.zip", fx.listings())
    df, log = data.load(d)
    return df, log


@pytest.fixture(scope="module")
def model(cars):
    df, _ = cars
    tr, ca, te = split(df)
    return PriceModel().fit(tr, ca, max_iter=200)


# ---------------------------------------------------------------- data
def test_loader_reads_zip_renames_and_cleans(cars):
    df, log = cars
    assert set(df.brand) == {"Ford", "Volkswagen", "Toyota", "Hyundai"}
    assert df.tax.notna().all()                       # hyundi's "tax(£)" column was renamed
    assert log["implausible_removed"] >= 2 and log["too_rare_removed"] >= 1
    assert df.year.max() <= 2020 and df.price.min() > 500 and "Other" not in set(df.transmission)
    assert (df.model == df.model.str.strip()).all() and "Ford Focus" in set(df.model_key)


# ---------------------------------------------------------------- needs parser
def test_parser_family_dog_london_commute():
    n, why = parse_lifestyle("Family of five in London with a dog, 40 mile commute, about £12k")
    assert n.budget == 12000 and n.seats == 5 and n.ulez
    assert set(n.sizes) == {"medium", "large", "xlarge"} and "estate" in n.bodies and "coupe" not in n.bodies
    assert 9000 <= n.annual_miles <= 14000
    assert any("London" in w for w in why)


def test_parser_does_not_read_mileage_as_budget():
    n, _ = parse_lifestyle("under 60,000 miles and under 9k")
    assert n.budget == 9000 and n.max_mileage == 60000
    n, _ = parse_lifestyle("I commute 60 miles a day, under 50k miles, £15,000")
    assert n.budget == 15000 and n.max_mileage == 50000 and n.annual_miles == 13800


def test_parser_new_driver_towing_and_conflicts():
    n, why = parse_lifestyle("just passed, first car, under 4k, LS6 2AB")
    assert n.budget == 4000 and n.max_engine == 1.4 and set(n.sizes) == {"city", "small"} and n.postcode == "LS6 2AB"
    n, why = parse_lifestyle("3 kids, tow a caravan, automatic, rural")
    assert n.seats == 5 and n.min_engine == 1.8 and n.transmission == "Automatic" and set(n.bodies) <= {"SUV", "estate", "pickup"}
    n, why = parse_lifestyle("first car but I have a big dog")
    assert any("conflicting" in w for w in why)      # small car vs big dog is reported, not silently resolved


def test_parser_low_mileage_avoids_diesel_and_seven_seats():
    n, _ = parse_lifestyle("retired, short trips only, need 7 seats for the grandchildren")
    assert "Diesel" not in n.fuels and n.seats == 7 and n.annual_miles == 5000


def test_postcode_validation():
    assert valid_postcode("sw1a1aa") and valid_postcode("LS6 2AB") and not valid_postcode("hello")


def test_llm_parser_uses_json_and_falls_back():
    reply = {"budget": 9000, "seats": 7, "fuels": ["Diesel", "Rocket"], "weights": {"space": 5}, "reasons": ["Seven seats for the family"]}
    client = types.SimpleNamespace(messages=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(content=[types.SimpleNamespace(text="Sure: " + json.dumps(reply))])))
    n, why, engine = parse_with_llm("big family", client)
    assert engine == "llm" and n.budget == 9000 and n.seats == 7 and n.fuels == ["Diesel"] and n.weights["space"] == 2
    broken = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **kw: 1 / 0))
    n, why, engine = parse_with_llm("family of five", broken)
    assert engine == "rules" and n.seats == 5 and "LLM unavailable" in why[-1]


# ---------------------------------------------------------------- costs
def test_fuel_cost_arithmetic_and_ulez():
    c = Costs(fuel_price_per_litre={"Petrol": 1.50, "Diesel": 1.50, "Hybrid": 1.5, "Other": 1.5})
    # 10,000 miles at 50 mpg = 200 gallons = 909.2 litres -> £1,363.83
    assert annual_fuel_cost([50], ["Petrol"], 10000, c)[0] == pytest.approx(1363.82, abs=0.05)
    assert list(ulez_ok(["Diesel", "Diesel", "Petrol", "Petrol"], [2015, 2016, 2005, 2006], c)) == [False, True, False, True]


# ---------------------------------------------------------------- price model
def test_price_model_beats_baselines_and_range_is_calibrated(cars, model):
    df, _ = cars
    _, _, te = split(df)
    y = te.price.to_numpy()
    gbm = metrics(y, model.predict(te))["mape"]
    base = metrics(y, MedianBaseline([["model_key", "year"], ["model_key"]]).fit(df.drop(te.index)).predict(te))["mape"]
    assert gbm < base
    fr = model.fair_range(te)
    cover = ((y >= fr.low) & (y <= fr.high)).mean()
    assert 0.65 <= cover <= 0.95
    # older and higher-mileage is worth less
    row = te.iloc[[0]]
    assert model.value_at(row, 3, 10000)[0] < model.predict(row)[0]


def test_deal_labels():
    assert deal_label(80, 100, 90, 110) == "Great price"
    assert deal_label(96, 100, 90, 110) == "Good price"
    assert deal_label(100, 100, 90, 110) == "Fair price"
    assert deal_label(105, 100, 90, 110) == "A bit high"
    assert deal_label(120, 100, 90, 110) == "Overpriced"


def test_unseen_model_still_predicts(model, cars):
    df, _ = cars
    row = df.iloc[[0]].copy()
    row["model_key"] = "Ford Unknown"
    assert np.isfinite(model.predict(row)).all()


# ---------------------------------------------------------------- recommender
def test_recommend_respects_requirements_and_explains(cars, model):
    df, _ = cars
    n = Needs(budget=12000, seats=7, postcode="LS6 2AB", annual_miles=12000)
    picks, info = recommend(df, n, model, min_listings=3)
    assert picks and {p.model_key for p in picks} <= {"Ford S-MAX", "Volkswagen Touran"}
    for p in picks:
        assert p.stats["price"] <= 12000 and p.stats["seats"] >= 7
        assert p.reasons and all(isinstance(r, str) for r in p.reasons)
        at = p.links[0]["url"]
        assert "autotrader.co.uk" in at and "postcode=LS62AB" in at and "price-to=12000" in at
        assert (p.examples.price_now <= 12000).all()
    assert len({p.model_key for p in picks}) == len(picks)          # one pick per model


def test_recommend_ulez_and_brand_cap(cars, model):
    df, _ = cars
    picks, _ = recommend(df, Needs(budget=40000, ulez=True), model, k=10, min_listings=3)
    brands = [p.brand for p in picks]
    assert max(brands.count(b) for b in set(brands)) <= 2
    base = with_attributes(df)
    d, steps = apply_filters(base, Needs(budget=40000, ulez=True), Costs())
    assert not ((d.fuel == "Diesel") & (d.year < 2016)).any()
    assert steps[-1][0] == "clean-air-zone compliant"


def test_empty_shortlist_suggests_what_to_relax(cars, model):
    df, _ = cars
    n = Needs(budget=3000, seats=7, fuels=["Hybrid"])
    picks, info = recommend(df, n, model, min_listings=3)
    assert picks == []
    sug = relax_suggestions(df, n, min_listings=3)
    assert sug and sug[0][1] > 0
    assert "Options" in why_empty(info, sug)


def test_priorities_change_the_ranking(cars, model):
    df, _ = cars
    cheap = Needs(budget=40000, weights={"running_costs": 2, "value": 0, "space": 0, "performance": 0, "newer": 0, "availability": 0, "low_depreciation": 0})
    roomy = Needs(budget=40000, weights={"running_costs": 0, "value": 0, "space": 2, "performance": 0, "newer": 0, "availability": 0, "low_depreciation": 0})
    a, _ = recommend(df, cheap, model, min_listings=3)
    b, _ = recommend(df, roomy, model, min_listings=3)
    assert a[0].model_key != b[0].model_key
    assert b[0].stats["size"] in ("large", "xlarge", "medium")


# ---------------------------------------------------------------- links
def test_links_encode_filters_and_drop_empty():
    ls = links.build_links("Mercedes-Benz", "A Class", postcode="ls6 2ab", radius=30, price_max=15000, year_from=2016, fuel="Diesel",
                           transmission="Semi-Auto")
    at = ls[0]["url"]
    assert "make=Mercedes-Benz" in at and "model=A+Class" in at and "postcode=LS62AB" in at and "radius=30" in at
    assert "transmission=Automatic" in at and "year-from=2016" in at and "maximum-mileage" not in at
    eb = next(x for x in ls if x["site"].startswith("eBay"))["url"]
    assert "_sacat=9801" in eb and "_udhi=15000" in eb and "_stpos=LS62AB" in eb
    no_pc = links.build_links("Ford", "Focus")
    assert "radius" not in no_pc[0]["url"] and "postcode" not in no_pc[0]["url"]


# ---------------------------------------------------------------- live APIs (offline)
def test_ebay_client_and_title_parsing(model, cars):
    df, _ = cars
    s = fx.FakeSession({"oauth2/token": fx.FakeResponse({"access_token": "T"}), "item_summary/search": fx.FakeResponse(fx.ebay_payload())})
    items = live.EbayClient("id", "secret", session=s).search("Ford", "Focus", 10000)
    assert s.calls[1][2]["headers"]["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_GB"
    assert "price:[..10000]" in s.calls[1][2]["params"]["filter"]
    a, b, c = items
    assert (a["year"], a["mileage"], a["engine_size"], a["price"]) == (2017, 42000, 1.5, 7995.0)
    assert (b["year"], b["mileage"], b["auction"]) == (2017, 18000, True)            # "67 reg" = Sept 2017
    assert c["year"] is None and c["price"] == 450.0
    priced = live.price_ebay(items, "Ford", "Focus", "Diesel", "Manual", model, df)
    assert "deal" in priced[0] and "deal" not in priced[2]


def test_plate_years():
    assert live.plate_year("17") == 2017 and live.plate_year("67") == 2017 and live.plate_year("51") == 2001


def test_mot_summary_flags_clocking_dangerous_and_recurring():
    s = live.summarise_mot(fx.mot_payload())
    assert s.tests == 6 and s.fails == 1 and s.dangerous == 1 and s.last_result == "PASS"
    assert s.mileage_drops and "49,000" in s.mileage_drops[0]
    assert s.recurring and "tyre" in s.recurring[0]
    assert any("DOWN" in f for f in s.flags) and any("dangerous" in f for f in s.flags)
    assert s.timeline[0]["date"] == "2018-04-03"                 # sorted by date
    assert live.summarise_mot({}).flags


def test_dvla_flags_and_client():
    assert any("does NOT meet" in f for f in live.dvla_flags(fx.dvla_payload()))                  # Euro 5 diesel
    assert any("ULEZ: meets" in f for f in live.dvla_flags(fx.dvla_payload(euroStatus="Euro 6d")))
    f = live.dvla_flags(fx.dvla_payload(euroStatus=None, fuelType="PETROL", yearOfManufacture=2009, taxStatus="SORN"))
    assert any("probably meets" in x for x in f) and any("SORN" in x for x in f)
    s = fx.FakeSession({"vehicle-enquiry": fx.FakeResponse(fx.dvla_payload())})
    v = live.DvlaClient("k", session=s).lookup("ab12 cde")
    assert v["make"] == "FORD" and s.calls[0][2]["json"] == {"registrationNumber": "AB12CDE"} and s.calls[0][2]["headers"]["x-api-key"] == "k"
    s404 = fx.FakeSession({})
    assert live.DvlaClient("k", session=s404).lookup("XX") == {}


def test_mot_client_auth_flow():
    s = fx.FakeSession({"login.microsoftonline.com": fx.FakeResponse({"access_token": "T"}), "history.mot.api.gov.uk": fx.FakeResponse(fx.mot_payload())})
    c = live.MotClient("id", "sec", "key", "tenant", session=s)
    assert c.configured and c.history("AB12 CDE")["make"] == "FORD"
    assert "tenant/oauth2/v2.0/token" in s.calls[0][1] and s.calls[0][2]["data"]["scope"] == "https://tapi.dvsa.gov.uk/.default"
    assert s.calls[1][1].endswith("/AB12CDE") and s.calls[1][2]["headers"]["X-API-Key"] == "key"
