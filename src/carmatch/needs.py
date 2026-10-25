"""What the buyer needs: a structured profile, plus a parser that turns a sentence about their life into one.

    parse_lifestyle("Family of five in London with a dog, 40 mile commute, about £12k")
      -> seats >= 5, bigger boot, ULEZ-compliant, ~9,200 miles a year, budget £12,000, and a list of what was understood.

The rule-based parser is transparent and testable: every change it makes comes with a plain-English reason, so the
buyer can see (and correct) what was assumed. `parse_with_llm` is an optional upgrade that asks an LLM for the same
JSON fields and falls back to the rules if anything goes wrong.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields, replace

BODIES = ["hatchback", "saloon", "estate", "SUV", "MPV", "coupe", "convertible", "pickup", "campervan"]
SIZES = ["city", "small", "medium", "large", "xlarge"]
FUELS = ["Petrol", "Diesel", "Hybrid", "Electric"]
PRIORITIES = {
    "running_costs": "Cheap to run (fuel and tax)",
    "value": "Good price for what it is",
    "space": "Space and practicality",
    "performance": "Power and performance",
    "newer": "Newer, lower mileage",
    "availability": "Easy to find (plenty for sale)",
    "low_depreciation": "Holds its value",
}
DEFAULT_WEIGHTS = {"running_costs": 1.0, "value": 1.0, "space": 0.5, "performance": 0.3, "newer": 0.7, "availability": 0.4,
                   "low_depreciation": 0.6}


@dataclass
class Needs:
    budget: float = 10000
    budget_min: float = 0
    seats: int = 0                       # minimum seats (0 = no requirement)
    bodies: list[str] = field(default_factory=list)     # allowed body types; empty = any
    sizes: list[str] = field(default_factory=list)      # allowed size classes; empty = any
    fuels: list[str] = field(default_factory=list)      # allowed fuels; empty = any
    transmission: str = "any"            # any | Manual | Automatic (Automatic includes semi-auto)
    annual_miles: int = 8000
    max_age: int | None = None           # years
    max_mileage: int | None = None
    max_engine: float | None = None      # litres (e.g. insurance for new drivers)
    min_engine: float | None = None      # litres (e.g. towing)
    ulez: bool = False                   # must meet ULEZ / Clean Air Zone standards
    years_owned: int = 3
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    postcode: str = ""
    radius: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Needs":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------- rule-based parser
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "a couple of": 2}
_NUM = r"(\d+|one|two|three|four|five|six|seven|eight|a couple of)"
CAZ_CITIES = ["london", "birmingham", "bristol", "bath", "bradford", "sheffield", "glasgow", "edinburgh", "aberdeen", "dundee",
              "portsmouth", "newcastle", "gateshead"]


def _n(s: str) -> int:
    return int(s) if s.isdigit() else WORDNUM[s]


def _money(s: str) -> float:
    s = s.lower().replace(",", "").replace("£", "").strip()
    mult = 1000 if s.endswith("k") else 1
    return float(s.rstrip("k")) * mult


def _restrict(n: "Needs", attr: str, allowed: list[str], why: list[str]) -> None:
    """Narrow an allowed-values list (empty = anything). Requirements combine by intersection; a clash is reported, not hidden."""
    cur = getattr(n, attr)
    if not cur:
        setattr(n, attr, list(allowed))
        return
    both = [v for v in cur if v in allowed]
    if both:
        setattr(n, attr, both)
    else:
        why.append(f"Note: conflicting {attr} requirements ({', '.join(cur)} vs {', '.join(allowed)}); kept the first")


def parse_lifestyle(text: str, base: Needs | None = None) -> tuple[Needs, list[str]]:
    """Returns the updated needs and the list of things understood (one reason per change)."""
    n = replace(base or Needs(), bodies=list((base or Needs()).bodies), sizes=list((base or Needs()).sizes),
                fuels=list((base or Needs()).fuels), weights=dict((base or Needs()).weights))
    t = " " + text.lower().replace("-", " ") + " "
    why: list[str] = []

    def bump(key, to, reason):
        if n.weights.get(key, 0) < to:
            n.weights[key] = to
        why.append(reason)

    # budget
    not_money = r"(?!\s?k?\s*(?:miles?|years?|seats?|kids|children|mpg|cc)\b)"
    m = re.search(r"(?:under|below|up to|max(?:imum)?|budget(?: of| is)?|around|about|spend|less than|no more than)\s*(£?\s?\d(?:[\d,.]*\d)?\s?k?)(?!\d|[,.]\d)\b" + not_money, t) \
        or re.search(r"(£\s?\d(?:[\d,.]*\d)?\s?k?)(?!\d|[,.]\d)\b", t)
    if m:
        val = _money(m.group(1))
        if val < 500:          # "£12" probably means £12k
            val *= 1000
        if 500 <= val <= 250000:
            n.budget = val
            why.append(f"Budget up to £{val:,.0f}")

    # people
    seats = 0
    if m := re.search(r"family of " + _NUM, t):
        seats = _n(m.group(1))
    if m := re.search(_NUM + r" (?:kids|children|sons|daughters|little ones|boys|girls)", t):
        kids = _n(m.group(1))
        adults = 1 if re.search(r"single (?:mum|dad|parent)", t) else 2
        seats = max(seats, adults + kids)
    elif re.search(r"\b(kids|children|toddler|baby|newborn|pram|buggy|car seats?)\b", t):
        seats = max(seats, 4)
        why.append("Children: needs rear doors and room for car seats")
        _restrict(n, "bodies", [b for b in BODIES if b not in ("coupe", "convertible", "pickup", "campervan")], why)
    if m := re.search(_NUM + r" ?seat(?:s|er)\b", t):
        seats = max(seats, _n(m.group(1)))
    if re.search(r"\b(grandparents|carpool|car pool|school run for)\b", t):
        seats = max(seats, 7)
    if seats:
        n.seats = max(n.seats, seats)
        why.append(f"At least {n.seats} seats" + (" (7-seat SUV or MPV)" if n.seats >= 6 else ""))
        if n.seats >= 5:
            bump("space", 1.0, "Space matters with a full car")

    # pets and gear
    if re.search(r"\b(dogs?|labrador|retriever|spaniel|puppy|pets?)\b", t):
        _restrict(n, "sizes", ["medium", "large", "xlarge"], why)
        _restrict(n, "bodies", ["estate", "SUV", "MPV", "hatchback"], why)
        bump("space", 1.0, "Dog: medium or bigger, with a boot it can ride in")
    if re.search(r"\b(bikes?|golf clubs|camping|surf|kayak|festival|tools|diy|skip runs?|band gear|equipment)\b", t):
        bump("space", 1.2, "Carries bulky gear: boot space weighted up")
        _restrict(n, "sizes", ["medium", "large", "xlarge"], why)

    # mileage
    miles = None
    if m := re.search(r"(\d[\d,]*)\s?(?:k\s)?miles? (?:a|per|each) (day|week|month|year)", t):
        v = float(m.group(1).replace(",", ""))
        if re.search(rf"{re.escape(m.group(1))}\s?k\s", t):
            v *= 1000
        per = m.group(2)
        mult = {"day": 230 if "commut" in t or "work" in t else 365, "week": 52, "month": 12, "year": 1}[per]
        miles = v * mult
    elif m := re.search(r"(\d[\d,]*)\s?mile (?:round trip |each way |daily )?commute", t):
        v = float(m.group(1).replace(",", ""))
        miles = v * (2 if "each way" in t else 1) * 230 + 2500
    elif re.search(r"\b(long commute|lots of motorway|motorway miles|drive a lot|high mileage|sales rep|travel for work)\b", t):
        miles = 15000
    elif re.search(r"\b(short trips|school run|local trips|pop to the shops|weekends only|barely drive|don't drive much|retired)\b", t):
        miles = 5000
    if miles:
        n.annual_miles = int(round(miles, -2))
        why.append(f"About {n.annual_miles:,} miles a year")
        if n.annual_miles >= 15000:
            bump("running_costs", 1.5, "High mileage: running costs weighted up (diesel and hybrids do well here)")
        elif n.annual_miles <= 6000 and not n.fuels:
            why.append("Low mileage: a diesel's economy rarely pays back, and short trips can clog its particulate filter")
            n.fuels = ["Petrol", "Hybrid", "Electric"]

    # where they drive
    city = [c for c in CAZ_CITIES if re.search(rf"\b{c}\b", t)]
    if city or re.search(r"\b(ulez|clean air zone|caz|low emission zone)\b", t):
        n.ulez = True
        why.append("Clean-air-zone compliant (ULEZ standard: petrol 2006 on, diesel late 2015 on)" + (f", for {city[0].title()}" if city else ""))
    if re.search(r"\b(city|town centre|narrow streets|tight parking|parallel park|on street parking|small garage)\b", t) and n.seats < 5:
        _restrict(n, "sizes", ["city", "small", "medium"], why)
        why.append("City driving: smaller, easier to park")
    if re.search(r"\b(rural|countryside|country lanes|farm|snow|muddy|off road|hills|highlands)\b", t):
        _restrict(n, "bodies", ["SUV", "estate", "pickup"], why)
        why.append("Rural roads: SUV or estate preferred (check for four-wheel drive on the advert)")

    # driver
    if re.search(r"\b(first car|new driver|just passed|learner|17 year old|18 year old|teen)\b", t):
        n.max_engine = min(n.max_engine or 9, 1.4)
        _restrict(n, "sizes", ["city", "small"], why)
        bump("running_costs", 1.5, "New driver: small engine (1.4L or less) to keep insurance down")
    if re.search(r"\b(tow|towing|caravan|horse box|horsebox|trailer|boat)\b", t):
        n.min_engine = max(n.min_engine or 0, 1.8)
        _restrict(n, "bodies", ["SUV", "estate", "pickup"], why)
        bump("performance", 1.2, "Towing: 1.8L or bigger engine, SUV/estate/pickup (check towing limit on the advert)")
    if re.search(r"\b(bad back|elderly|mobility|easy to get in|high seating|sit up high|wheelchair)\b", t):
        _restrict(n, "bodies", ["SUV", "MPV"], why)
        why.append("Easy access: higher seating (SUV or MPV)")

    # preferences
    if re.search(r"\b(automatic|auto only|no gears|not manual|can't drive manual)\b", t) or re.search(r"\bauto\b", t):
        n.transmission = "Automatic"
        why.append("Automatic gearbox")
    elif re.search(r"\b(manual|stick shift)\b", t):
        n.transmission = "Manual"
        why.append("Manual gearbox")
    if re.search(r"\b(electric|ev|zero emission)\b", t) and not re.search(r"\bno (electric|ev)\b", t):
        n.fuels = ["Electric"] + (["Hybrid"] if "hybrid" in t else [])
        why.append("Electric" + (" or hybrid" if "hybrid" in t else "") + " (you'll want somewhere to charge)")
    elif "hybrid" in t:
        n.fuels = ["Hybrid"]
        why.append("Hybrid")
    elif re.search(r"\bno diesel|not diesel|avoid diesel\b", t):
        n.fuels = [f for f in (n.fuels or FUELS) if f != "Diesel"]
        why.append("No diesel")
    elif "diesel" in t:
        n.fuels = ["Diesel"]
        why.append("Diesel")
    if re.search(r"\b(cheap to run|economical|fuel efficient|good mpg|low running costs|save money|tight budget|cost of living)\b", t):
        bump("running_costs", 2.0, "Running costs matter most")
    if re.search(r"\b(fun|sporty|fast|quick|powerful|performance|weekend toy)\b", t):
        bump("performance", 1.5, "Something fun to drive: power weighted up")
    if re.search(r"\b(reliable|reliability|no hassle|peace of mind|won't break)\b", t):
        bump("newer", 1.5, "Reliability: newer and lower-mileage cars weighted up (the data has no breakdown records, so check the MOT history)")
    if re.search(r"\b(keep it (?:for )?(?:a long time|years)|long term|resale|sell it on)\b", t):
        bump("low_depreciation", 1.5, "Resale value weighted up")
    if m := re.search(r"(?:keep it|own it) for " + _NUM + r" years?", t):
        n.years_owned = max(1, min(10, _n(m.group(1))))
        why.append(f"Owned for {n.years_owned} years (for running-cost and depreciation estimates)")
    if m := re.search(r"(?:no older than|newer than|less than|under) " + _NUM + r" years? old", t):
        n.max_age = _n(m.group(1))
        why.append(f"No older than {n.max_age} years")
    if m := re.search(r"(?:under|less than|below) (\d[\d,]*k?) miles\b(?! (?:a|an|per|each) )", t):
        n.max_mileage = int(_money(m.group(1)))
        why.append(f"Under {n.max_mileage:,} miles on the clock")
    if m := re.search(r"\b([a-z]{1,2}\d[a-z\d]?\s?\d[a-z]{2})\b", t):
        n.postcode = m.group(1).upper()
        why.append(f"Near {n.postcode}")

    # contradictions worth flagging rather than silently resolving
    if n.seats >= 6 and n.sizes and not set(n.sizes) & {"medium", "large", "xlarge"}:
        why.append("Note: 6+ seats and a small car pull in opposite directions; seats were kept as the hard requirement")
        n.sizes = []
    if n.seats >= 6:
        n.sizes = [s for s in n.sizes if s in ("medium", "large", "xlarge")]
    return n, why


def _norm_postcode(pc: str) -> str:
    pc = re.sub(r"\s+", "", pc or "").upper()
    return pc[:-3] + " " + pc[-3:] if len(pc) >= 5 else pc


POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$")


def valid_postcode(pc: str) -> bool:
    return bool(POSTCODE_RE.match(_norm_postcode(pc)))


# ---------------------------------------------------------------- optional LLM parser
LLM_PROMPT = """You turn a UK car buyer's description of their life into search requirements.
Reply with JSON only, using any of these keys (omit what the text doesn't imply):
budget (number, GBP), seats (min seats), bodies (subset of {bodies}), sizes (subset of {sizes}),
fuels (subset of {fuels}), transmission ("Manual" | "Automatic"), annual_miles (number), max_age (years),
max_mileage, max_engine (litres), min_engine (litres), ulez (true if they live or drive in a UK clean air zone),
weights (object; keys from {weights}; values 0-2 for how much each matters), reasons (list of short strings
explaining each requirement in plain English).
Text: {text}"""


def parse_with_llm(text: str, client=None, model: str | None = None, base: Needs | None = None) -> tuple[Needs, list[str], str]:
    """Ask an LLM for the requirements; returns (needs, reasons, engine). Falls back to rules on any failure.

    `client` is an anthropic.Anthropic() instance (pip install anthropic; set ANTHROPIC_API_KEY)."""
    import os
    rules, rules_why = parse_lifestyle(text, base)
    if client is None:
        return rules, rules_why, "rules"
    try:
        msg = client.messages.create(
            model=model or os.environ.get("CARMATCH_LLM_MODEL", "claude-haiku-4-5-20251001"), max_tokens=800,
            messages=[{"role": "user", "content": LLM_PROMPT.format(bodies=BODIES, sizes=SIZES, fuels=FUELS,
                                                                     weights=list(PRIORITIES), text=text)}])
        raw = msg.content[0].text
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception as e:  # noqa: BLE001 - any failure means "use the rules"
        return rules, rules_why + [f"(LLM unavailable: {type(e).__name__}; used the built-in rules)"], "rules"
    n = replace(base or Needs())
    n.weights = dict(n.weights)
    ok = {"budget": float, "seats": int, "annual_miles": int, "max_age": int, "max_mileage": int, "max_engine": float,
          "min_engine": float, "ulez": bool}
    for k, typ in ok.items():
        if k in data and data[k] is not None:
            try:
                setattr(n, k, typ(data[k]))
            except (TypeError, ValueError):
                pass
    for k, allowed in (("bodies", BODIES), ("sizes", SIZES), ("fuels", FUELS)):
        if isinstance(data.get(k), list):
            setattr(n, k, [v for v in data[k] if v in allowed])
    if data.get("transmission") in ("Manual", "Automatic"):
        n.transmission = data["transmission"]
    for k, v in (data.get("weights") or {}).items():
        if k in PRIORITIES and isinstance(v, (int, float)):
            n.weights[k] = float(max(0, min(2, v)))
    n.postcode = rules.postcode or n.postcode
    reasons = [str(r) for r in data.get("reasons", [])][:12] or rules_why
    return n, reasons, "llm"
