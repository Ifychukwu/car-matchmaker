"""Car Matchmaker: tell it about your life, get a shortlist of used cars that fit, and links to ones for sale near you.

    pip install -r requirements.txt
    streamlit run app.py
Needs the Kaggle zip in data/raw/ (see data/README.md). The price model trains on first run (~15 seconds) and is cached.
"""
from __future__ import annotations

import html
import os

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from carmatch import live  # noqa: E402
from carmatch.catalogue import AMBIGUOUS, CATALOGUE  # noqa: E402
from carmatch.costs import Costs  # noqa: E402
from carmatch.data import ROOT, load  # noqa: E402
from carmatch.links import build_links  # noqa: E402
from carmatch.needs import BODIES, PRIORITIES, SIZES, Needs, parse_with_llm, valid_postcode  # noqa: E402
from carmatch.price_model import MODELS, PriceModel, deal_label, evaluate  # noqa: E402
from carmatch.recommend import recommend, relax_suggestions, why_empty, with_attributes  # noqa: E402

st.set_page_config(page_title="Car Matchmaker", page_icon="🚗", layout="wide")
DEAL_COLOUR = {"Great price": "#1baf7a", "Good price": "#4c9e74", "Fair price": "#6b6a66", "A bit high": "#c97a1c", "Overpriced": "#c43d2e"}
EXAMPLES = [
    "Family of five in London with a dog, 40 mile commute, about £12k",
    "Just passed my test, first car, cheap to run, under £5k",
    "Three kids, we tow a small caravan, rural Wales, automatic, budget £20,000",
    "Retired, short trips, bad back so I want to sit up high, around £10k",
]


@st.cache_resource(show_spinner="Loading 97,000 adverts and training the price model (first run only)...")
def resources():
    df, log = load()
    df = with_attributes(df)
    path = MODELS / "price_model.pkl"
    try:
        model = PriceModel.load(path)
        model.predict(df.head(2))
    except Exception:  # noqa: BLE001 - missing or built with another sklearn: retrain
        model, _, _ = evaluate(df)
        model.save(path)
    return df, log, model


try:
    df, log, pm = resources()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

costs = Costs.load()
ss = st.session_state

# ---------------------------------------------------------------- sidebar: assumptions
with st.sidebar:
    st.markdown("### Cost assumptions")
    st.caption("Used for running costs and the cost of ownership. Edit to today's prices.")
    costs.fuel_price_per_litre["Petrol"] = costs.fuel_price_per_litre["Hybrid"] = costs.fuel_price_per_litre["Other"] = \
        st.number_input("Petrol, £ per litre", 0.5, 3.0, float(costs.fuel_price_per_litre["Petrol"]), step=0.01)
    costs.fuel_price_per_litre["Diesel"] = st.number_input("Diesel, £ per litre", 0.5, 3.0, float(costs.fuel_price_per_litre["Diesel"]), step=0.01)
    costs.real_world_mpg_factor = st.slider("Real-world mpg as % of the listed figure", 60, 100, int(costs.real_world_mpg_factor * 100)) / 100
    costs.market_index = st.slider("Price level vs the 2020 data (%)", 70, 160, int(costs.market_index * 100),
                                   help="The adverts are from 2020. If used prices are higher today, raise this; budgets are compared "
                                        "against data prices × this factor.") / 100
    st.markdown("### Optional live data")
    st.caption("Free keys, kept for this session only. See the README for where to get them.")
    ebay_id = st.text_input("eBay client ID", os.environ.get("EBAY_CLIENT_ID", ""), type="password")
    ebay_secret = st.text_input("eBay client secret", os.environ.get("EBAY_CLIENT_SECRET", ""), type="password")
    dvla_key = st.text_input("DVLA API key", os.environ.get("DVLA_API_KEY", ""), type="password")
    with st.expander("MOT history API keys"):
        mot = {k: st.text_input(lbl, os.environ.get(env, ""), type="password") for k, lbl, env in [
            ("client_id", "Client ID", "MOT_CLIENT_ID"), ("client_secret", "Client secret", "MOT_CLIENT_SECRET"),
            ("api_key", "API key", "MOT_API_KEY"), ("tenant_id", "Tenant ID", "MOT_TENANT_ID")]}

st.title("🚗 Car Matchmaker")
st.caption(f"Recommends used cars from {log['rows']:,} real UK adverts ({log['models']} models, 9 brands), explains every pick, "
           "prices it against a fair-value model, and links to ones for sale near you today.")

tab_find, tab_price, tab_ebay, tab_reg, tab_model, tab_about = st.tabs(
    ["🔎 Find my car", "🏷️ Is this a good price?", "🛒 Live eBay listings", "🔍 Check a number plate", "📊 How accurate is it?", "ℹ️ How it works"])


def esc(x) -> str:
    return html.escape(str(x))


def money(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"£{x:,.0f}"


# ---------------------------------------------------------------- 1. find my car
with tab_find:
    st.markdown("#### 1. Tell me about your life")
    ex = st.selectbox("Try an example", ["(write your own)"] + EXAMPLES)
    text = st.text_area("Who rides with you, where you drive, what you carry, your budget, anything you care about",
                        "" if ex == "(write your own)" else ex, height=90)
    use_llm = st.toggle("Use an AI model to read it (needs ANTHROPIC_API_KEY; otherwise the built-in rules are used)", False)
    if st.button("Understand my needs", type="primary") and text.strip():
        client = None
        if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                client = anthropic.Anthropic()
            except ImportError:
                st.warning("pip install anthropic to use the AI reader; using the built-in rules.")
        n, why, engine = parse_with_llm(text, client)
        ss["needs"], ss["why"], ss["engine"] = n.to_dict(), why, engine
        ss["form_v"] = ss.get("form_v", 0) + 1
    n0 = Needs.from_dict(ss["needs"]) if "needs" in ss else Needs()
    if ss.get("why"):
        st.markdown("**What I understood**" + (" (AI reader)" if ss.get("engine") == "llm" else "") + ":")
        st.markdown("\n".join(f"- {w}" for w in ss["why"]))

    st.markdown("#### 2. Check and adjust")
    v = ss.get("form_v", 0)
    a, b, c3 = st.columns(3)
    budget = a.number_input("Budget (£)", 500, 150000, int(n0.budget), step=500, key=f"budget{v}")
    seats = a.selectbox("Minimum seats", [0, 2, 4, 5, 7, 8], index=[0, 2, 4, 5, 7, 8].index(n0.seats) if n0.seats in [0, 2, 4, 5, 7, 8] else 0,
                        format_func=lambda s: "Any" if s == 0 else f"{s}+", key=f"seats{v}")
    annual = a.number_input("Miles a year", 1000, 50000, int(n0.annual_miles), step=500, key=f"miles{v}")
    bodies = b.multiselect("Body types (empty = any)", BODIES, n0.bodies, key=f"bodies{v}")
    sizes = b.multiselect("Sizes (empty = any)", SIZES, n0.sizes, key=f"sizes{v}")
    fuel_default = [f for f in n0.fuels if f != "Electric"] or (["Hybrid"] if "Electric" in n0.fuels else [])
    fuels = b.multiselect("Fuel (empty = any)", ["Petrol", "Diesel", "Hybrid"], fuel_default, key=f"fuels{v}")
    if "Electric" in n0.fuels:
        b.info("The 2020 adverts contain no electric cars, so EVs can't be recommended here. Hybrids are shown instead; "
               "the dealer links can still be switched to electric on each site.")
    gear = c3.radio("Gearbox", ["any", "Manual", "Automatic"], index=["any", "Manual", "Automatic"].index(n0.transmission), horizontal=True, key=f"gear{v}")
    max_age = c3.number_input("Max age in years (0 = any)", 0, 25, int(n0.max_age or 0), key=f"age{v}")
    max_miles = c3.number_input("Max mileage (0 = any)", 0, 200000, int(n0.max_mileage or 0), step=5000, key=f"mm{v}")
    ulez = c3.checkbox("Must meet ULEZ / clean-air-zone rules", n0.ulez, key=f"ulez{v}")
    a, b, c3 = st.columns(3)
    postcode = a.text_input("Your postcode (for dealer links)", n0.postcode, key=f"pc{v}")
    radius = b.selectbox("Search radius (miles)", [10, 25, 50, 100, 200], index=2, key=f"rad{v}")
    years = c3.slider("Years you'll keep it", 1, 8, int(n0.years_owned), key=f"yrs{v}")
    if postcode and not valid_postcode(postcode):
        st.warning("That doesn't look like a UK postcode; links will search nationwide.")
        postcode = ""
    with st.expander("What matters most to you? (priorities)", expanded=False):
        cols = st.columns(4)
        weights = {k: cols[i % 4].slider(lbl, 0.0, 2.0, float(n0.weights.get(k, 0)), 0.1, key=f"w_{k}{v}") for i, (k, lbl) in enumerate(PRIORITIES.items())}

    needs = Needs(budget=budget, seats=seats, bodies=bodies, sizes=sizes, fuels=fuels, transmission=gear, annual_miles=annual,
                  max_age=max_age or None, max_mileage=max_miles or None, max_engine=n0.max_engine, min_engine=n0.min_engine, ulez=ulez,
                  years_owned=years, weights=weights, postcode=postcode, radius=radius)
    if n0.max_engine or n0.min_engine:
        st.caption(f"Engine size limit from your description: {n0.min_engine or 0}–{n0.max_engine or '∞'} litres.")

    st.markdown("#### 3. Your shortlist")
    picks, info = recommend(df, needs, pm, costs)
    if not picks:
        st.warning(why_empty(info, relax_suggestions(df, needs, costs)))
    else:
        st.caption(f"{info['matching_listings']:,} adverts across {info['models_considered']} models meet every requirement. "
                   "Scores rank the models against each other on your priorities. Prices are from the 2020 data"
                   + (f" × {costs.market_index:.0%}" if costs.market_index != 1 else "") + "; the links show today's.")
    for i, p in enumerate(picks, 1):
        s = p.stats
        with st.container(border=True):
            h, m1, m2, m3, m4 = st.columns([3, 1, 1, 1, 1])
            h.markdown(f"### {i}. {esc(p.model_key)} <span style='font-size:0.6em;color:#6b6a66'>{esc(p.fuel.lower())} · "
                       f"{esc(s['size'])} {esc(s['body'])} · {int(s['seats']) if pd.notna(s['seats']) else '?'} seats</span>",
                       unsafe_allow_html=True)
            h.progress(min(1.0, p.score), text=f"Match {p.score:.0%}")
            m1.metric("Typical price", money(s["price"]))
            m2.metric("Fuel + tax / year", money(s["running"]))
            m3.metric("Cost / year to own", money(s["cost_year"]), help=f"Depreciation over {years} years plus fuel and tax, per year")
            m4.metric("Typical age · miles", f"{s['age']:.0f} y · {s['mileage'] / 1000:.0f}k")
            st.markdown("**Why it fits:** " + " · ".join(p.reasons))
            if p.tradeoffs:
                st.markdown("**Trade-offs:** " + " · ".join(p.tradeoffs))
            lc = st.columns(len(p.links))
            for col, ln in zip(lc, p.links):
                col.link_button(f"Find on {ln['site']}", ln["url"], use_container_width=True)
            with st.expander(f"Best-value examples from the data ({int(s['n'])} adverts match)"):
                exs = p.examples.copy()
                lo, hi = np.exp(-pm.q80), np.exp(pm.q80)
                exs["deal"] = [deal_label(r.price_now, r.fair, r.fair * lo, r.fair * hi) for r in exs.itertuples()]
                st.dataframe(exs.rename(columns={"price_now": "price", "fair": "fair value", "cost_per_year": "cost/yr to own",
                                                "engine_size": "engine (L)", "deal_ratio": "price ÷ fair"}),
                             hide_index=True, use_container_width=True,
                             column_config={"price": st.column_config.NumberColumn(format="£%d"), "fair value": st.column_config.NumberColumn(format="£%d"),
                                            "cost/yr to own": st.column_config.NumberColumn(format="£%d"), "price ÷ fair": st.column_config.NumberColumn(format="%.2f")})

# ---------------------------------------------------------------- 2. price check
with tab_price:
    st.markdown("Paste the details from an advert. The model estimates a fair price from 97,000 comparable adverts and shows how much it will be worth later.")
    brands = sorted(df.brand.unique())
    a, b, c3, d = st.columns(4)
    brand = a.selectbox("Make", brands, index=brands.index("Ford") if "Ford" in brands else 0)
    models = sorted(df[df.brand == brand].model.unique())
    model_name = b.selectbox("Model", models, index=models.index("Focus") if "Focus" in models else 0)
    sub = df[(df.brand == brand) & (df.model == model_name)]
    fuel = c3.selectbox("Fuel", sorted(sub.fuel.unique()))
    trans = d.selectbox("Gearbox", sorted(sub.transmission.unique()))
    a, b, c3, d = st.columns(4)
    year = a.number_input("Registration year", 1996, 2020, int(sub.year.median()), help="The data runs to 2020; for a newer car use 2020.")
    miles = b.number_input("Mileage", 0, 300000, int(sub.mileage.median()), step=1000)
    eng = c3.number_input("Engine size (litres)", 0.6, 6.5, float(sub.engine_size.median() if sub.engine_size.notna().any() else 1.5), step=0.1)
    asking = d.number_input("Asking price (£)", 0, 200000, 0, step=100, help="Leave at 0 to just see the fair value")
    row = pd.DataFrame([{"brand": brand, "model": model_name, "model_key": f"{brand} {model_name}", "fuel": fuel, "transmission": trans,
                         "year": year, "age": pm.as_of_year - year, "mileage": miles, "engine_size": eng,
                         "mpg": sub[sub.fuel == fuel].mpg.median(), "tax": sub[sub.fuel == fuel].tax.median()}])
    fr = pm.fair_range(row).iloc[0] * costs.market_index
    m1, m2, m3 = st.columns(3)
    m1.metric("Fair value", money(fr.fair))
    m2.metric("Fair range (80%)", f"{money(fr.low)} – {money(fr.high)}")
    if asking:
        lab = deal_label(asking, fr.fair, fr.low, fr.high)
        m3.metric("Verdict", lab, f"{asking / fr.fair - 1:+.0%} vs fair value", delta_color="inverse")
    yrs = np.arange(0, 7)
    proj = [pm.value_at(row, y, 8000)[0] * costs.market_index for y in yrs]
    st.markdown(f"**Expected value over the next 6 years** at 8,000 miles a year (losing about {1 - proj[3] / proj[0]:.0%} in 3 years):")
    st.line_chart(pd.DataFrame({"value (£)": proj}, index=[f"+{y}y" for y in yrs]), height=240)
    comp = sub[(sub.fuel == fuel) & (sub.year.between(year - 1, year + 1))]
    st.caption(f"{len(comp)} comparable adverts in the data ({fuel}, {year - 1}–{year + 1}): median {money(comp.price.median() * costs.market_index)}. "
               f"Prices are 2020 asking prices{' × ' + format(costs.market_index, '.0%') if costs.market_index != 1 else ''}.")
    for ln in build_links(brand, model_name, price_max=None, fuel=fuel):
        st.markdown(f"- [{ln['site']}: {brand} {model_name} for sale today]({ln['url']})")

# ---------------------------------------------------------------- 3. live eBay
with tab_ebay:
    st.markdown("Today's listings from eBay's official Browse API, each checked against the fair-price model "
                "(where the title gives the year and mileage).")
    ebay = live.EbayClient(ebay_id, ebay_secret)
    a, b, c3, d = st.columns(4)
    e_brand = a.selectbox("Make ", sorted(df.brand.unique()))
    e_model = b.selectbox("Model ", sorted(df[df.brand == e_brand].model.unique()))
    e_fuel = c3.selectbox("Fuel ", ["Petrol", "Diesel", "Hybrid"])
    e_max = d.number_input("Max price (£)", 0, 200000, 15000, step=500)
    if not ebay.configured:
        st.info("Add an eBay client ID and secret in the sidebar to see live listings (free at developer.ebay.com). "
                "Meanwhile, this opens the same search on eBay:")
        st.link_button("Open on eBay", build_links(e_brand, e_model, price_max=e_max)[1]["url"])
    elif st.button("Search eBay"):
        try:
            items = live.price_ebay(ebay.search(e_brand, e_model, e_max or None), e_brand, e_model, e_fuel, "Manual", pm, df)
            items = [{**it, "fair": it.get("fair", np.nan) * costs.market_index} for it in items]
            st.caption(f"{len(items)} listings; {sum('deal' in i for i in items)} could be priced from their titles. "
                       "Fair values assume a manual gearbox and this model's typical mpg; check the advert.")
            for it in sorted(items, key=lambda x: (x.get("price", 0) / x["fair"]) if x.get("fair") == x.get("fair") else 9):
                with st.container(border=True):
                    l, r = st.columns([1, 4])
                    if it["image"]:
                        l.image(it["image"], use_container_width=True)
                    deal = it.get("deal")
                    badge = f"<span style='background:{DEAL_COLOUR[deal]};color:white;padding:2px 8px;border-radius:8px;font-size:0.8em'>{deal}</span>" if deal else ""
                    r.markdown(f"**[{esc(it['title'])}]({it['url']})** {badge}", unsafe_allow_html=True)
                    r.caption(" · ".join(x for x in [money(it["price"]) + (" (current bid)" if it["auction"] else ""),
                                                      f"fair ≈ {money(it['fair'])}" if deal else "", it["location"],
                                                      f"{it['year']}" if it["year"] else "", f"{it['mileage']:,} miles" if it["mileage"] else ""] if x))
        except Exception as e:  # noqa: BLE001
            st.error(f"eBay request failed: {e}. Check the keys are a Production keyset.")

# ---------------------------------------------------------------- 4. number plate
with tab_reg:
    st.markdown("Before you buy: check the tax, MOT, clean-air-zone standard and full MOT history (mileage at every test, "
                "failures and advisories) using the DVLA and DVSA APIs.")
    reg = st.text_input("Number plate", placeholder="e.g. AB12 CDE")
    dv, mc = live.DvlaClient(dvla_key), live.MotClient(**mot)
    if not (dv.configured or mc.configured):
        st.info("Add a DVLA API key and/or MOT history keys in the sidebar (both free, from the DVLA developer portal and "
                "documentation.history.mot.api.gov.uk). Without keys you can check the same plate on gov.uk:")
        st.link_button("GOV.UK MOT history", "https://www.gov.uk/check-mot-history")
    elif reg and st.button("Check this car"):
        if dv.configured:
            try:
                v = dv.lookup(reg)
                if v:
                    cols = st.columns(4)
                    cols[0].metric("Make · colour", f"{v.get('make', '?').title()} · {str(v.get('colour', '')).title()}")
                    cols[1].metric("Year · fuel", f"{v.get('yearOfManufacture', '?')} · {str(v.get('fuelType', '')).title()}")
                    cols[2].metric("Tax", v.get("taxStatus", "?"), v.get("taxDueDate", ""), delta_color="off")
                    cols[3].metric("MOT", v.get("motStatus", "?"), v.get("motExpiryDate", ""), delta_color="off")
                for f in live.dvla_flags(v):
                    (st.success if "meets the" in f and "NOT" not in f else st.warning)(f)
            except Exception as e:  # noqa: BLE001
                st.error(f"DVLA lookup failed: {e}")
        if mc.configured:
            try:
                s = live.summarise_mot(mc.history(reg))
                cols = st.columns(4)
                cols[0].metric("MOT tests", s.tests, f"{s.fails} failed", delta_color="off")
                cols[1].metric("Last recorded mileage", f"{s.last_mileage:,}" if s.last_mileage else "–")
                cols[2].metric("Miles a year", f"{s.miles_per_year:,.0f}" if s.miles_per_year else "–")
                cols[3].metric("Advisories", s.advisories, f"{s.dangerous} dangerous", delta_color="off")
                for f in s.flags:
                    st.warning(f)
                if not s.flags:
                    st.success("No red flags in the MOT history.")
                if s.timeline:
                    tl = pd.DataFrame(s.timeline)
                    st.line_chart(tl.set_index("date")["mileage"], height=220)
                    st.dataframe(tl, hide_index=True, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"MOT history lookup failed: {e}")

# ---------------------------------------------------------------- 5. accuracy
with tab_model:
    import json
    mfile = ROOT / "reports" / "metrics.json"
    if mfile.exists():
        mm = json.loads(mfile.read_text())
        o = mm["overall"]
        st.markdown(f"Scored once on {mm['sizes']['test']:,} held-out adverts the model never saw.")
        c = st.columns(4)
        c[0].metric("Average error (MAPE)", f"{o['gbm']['mape']:.1f}%", f"{o['gbm']['mape'] - o['model_year_median']['mape']:+.1f} pts vs model+year median", delta_color="inverse")
        c[1].metric("Average error (£)", money(o["gbm"]["mae"]), f"{o['gbm']['mae'] - o['model_year_median']['mae']:+,.0f} vs baseline", delta_color="inverse")
        c[2].metric("Within 10% of the asking price", f"{o['gbm']['within_10pct']:.0f}%", f"{o['gbm']['within_10pct'] - o['model_year_median']['within_10pct']:+.0f} pts")
        c[3].metric("Fair-range coverage (target 80%)", f"{mm['fair_range']['test_coverage']:.1f}%", f"±{mm['fair_range']['half_width_pct']:.0f}% wide", delta_color="off")
        for fig in ["accuracy_vs_baselines.png", "predicted_vs_actual.png", "error_by_band_and_brand.png", "depreciation_curves.png"]:
            p = ROOT / "reports" / "figures" / fig
            if p.exists():
                st.image(str(p))
    else:
        st.info("Run `python -m carmatch.price_model` and `python -m carmatch.report` to produce the evaluation.")

with tab_about:
    st.markdown(f"""
**1. Understand the buyer.** A sentence about your life becomes requirements (seats, size, body, fuel, gearbox, budget,
mileage, clean-air zone, engine limits) and priorities. Every assumption is listed so you can correct it, and clashing
needs (a big dog and a city car) are flagged, not silently resolved. An optional AI reader does the same job with an LLM.

**2. Filter real adverts.** {log['rows']:,} UK adverts (Kaggle "100,000 UK Used Car Data set", 2020). Body type, seats and
size come from a hand-built catalogue covering {sum(k in CATALOGUE for k in df.model_key.unique())} of {df.model_key.nunique()} models;
{len(AMBIGUOUS)} labels in the data mix different cars and are never recommended.

**3. Price and cost everything.** A gradient-boosted model predicts each advert's fair price and its value after the years
you'll keep it. Cost per year = depreciation + fuel + tax.

**4. Rank and explain.** Models are scored on your priorities relative to everything else that fits, one pick per model and
at most two per brand, each with the reasons it ranked and what you give up.

**5. Find one for sale.** Links open AutoTrader, eBay and Gumtree already filtered to the model, budget, fuel, gearbox and
your postcode, so you see today's stock at real dealers near you.

**Limits.** The adverts are from 2020 and contain no electric cars; prices are relative guides, so use the price-level slider
or compare the "price ÷ fair" ratio. Insurance, servicing and reliability are not modelled: check the MOT history.
""")
