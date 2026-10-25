# Methods

## 1. From a sentence to requirements (`needs.py`)
Regular-expression rules map phrases to fields of a `Needs` profile and append a plain-English reason for each:

| Phrase | Effect |
|---|---|
| "family of five", "3 kids" (+2 adults; +1 if "single parent"), "7 seater", "grandparents" | minimum seats |
| "dog", "bikes", "camping", "golf clubs" | size medium or bigger; space weighted up |
| "40 mile commute", "60 miles a day" (×230 work days or ×365), "short trips" | yearly miles |
| London and other clean-air-zone cities, "ULEZ" | ULEZ: petrol 2006 on, diesel 2016 on |
| "first car", "just passed" | engine ≤ 1.4L, city/small, running costs weighted up |
| "tow", "caravan" | engine ≥ 1.8L, SUV/estate/pickup |
| "bad back", "elderly" | SUV or MPV |
| "under 50k miles", "no older than 5 years", "£12k", postcodes | hard limits |

Restrictions combine by **intersection**; if two needs leave nothing (a new driver's small car and a big dog), the
clash is reported instead of silently dropped. Low yearly mileage removes diesel (a particulate filter needs long runs).
The optional LLM reader asks for the same JSON fields and validates every value against the allowed lists; any
failure falls back to the rules.

## 2. Filtering
Budget (data price × market index), seats, body, size, fuel, gearbox (automatic includes semi-auto), age, mileage,
engine limits, ULEZ by registration year. The count each filter removes is kept, and when nothing fits,
`relax_suggestions` re-runs the search with each requirement loosened (budget +25%, others dropped) and reports how
many models each change would open up.

## 3. Costing every advert (`costs.py`, `price_model.py`)
* fuel per year = miles ÷ (listed mpg × real-world factor) × 4.54609 litres per gallon × £ per litre
* tax per year = the advert's road tax
* resale = model prediction at age + N years and mileage + N × yearly miles
* cost per year = (price − resale + N × (fuel + tax)) ÷ N
* deal ratio = price ÷ fair price

## 4. Scoring and explaining (`recommend.py`)
Adverts are grouped by model and fuel (at least 5 adverts). Each group gets percentile-rank scores within the shortlist:

| Priority | Measured by |
|---|---|
| running costs | median fuel + tax per year (lower is better) |
| value | median deal ratio and share of adverts ≥3% under fair value |
| space | size class + seats ÷ 7 + bonus for MPV/estate/SUV |
| performance | median engine size |
| newer | median age and mileage |
| availability | log of the number of matching adverts |
| holds its value | median depreciation % (lower is better) |

Total = weighted average with the buyer's weights, plus a 0.05 nudge towards the lowest cost per year to break ties.
Picks: one per model, at most two per brand. Reasons are the top three weighted factors that scored ≥ 0.6, written
with the group's own numbers; trade-offs are weighted factors (weight ≥ 0.5) that scored ≤ 0.3.

## 5. Links (`links.py`)
URL templates per site in `config/dealers.yaml`, filled with make, model, postcode, radius, maximum price, fuel,
gearbox and, only when the buyer asked for a maximum age or ULEZ, a minimum year. Data years aren't used for links
because the data's prices are from 2020.

## 6. Live checks (`live.py`)
* **eBay Browse API**: OAuth client-credentials token, then `item_summary/search` in category 9801 (Cars) on EBAY_GB.
  Year ("2017" or "67 reg"), mileage ("42,000 miles", "18k miles") and engine size are read from the title; items with
  a year and mileage get a fair price and deal label.
* **DVLA Vehicle Enquiry**: POST the plate with an `x-api-key`; ULEZ from `euroStatus` (Euro 4 petrol, Euro 6 diesel),
  else judged by year and marked as such.
* **DVSA MOT history**: Microsoft Entra client-credentials token (scope `https://tapi.dvsa.gov.uk/.default`) plus
  `X-API-Key`. Flags: mileage lower than at an earlier test, dangerous defects, failing the latest test, the same
  advisory on two or more tests, and over 20,000 miles a year.
