# Model card: used-car fair-price model

**What it does.** Estimates the fair asking price of a UK used car from its model, age, mileage, engine size, fuel,
gearbox, listed mpg and road tax, with an 80% fair range. It feeds the deal verdicts, resale estimates (value after N
more years and miles) and the cost of ownership in the recommender.

**Model.** scikit-learn `HistGradientBoostingRegressor` on log(price); native categorical features (model, brand, fuel,
gearbox); 800 iterations, learning rate 0.06, 63 leaves, min 10 samples per leaf, L2 0.1, seed 42.

**Data.** Kaggle "100,000 UK Used Car Data set": 97,647 adverts after cleaning, 9 brands (Audi, BMW, Ford, Hyundai,
Mercedes-Benz, Skoda, Toyota, Vauxhall, Volkswagen), registration years 1996–2020, scraped in 2020.

**Evaluation.** Frozen random split: 68,352 train, 9,765 calibration, 19,530 test (scored once).

| | MAPE | Median APE | MAE | Within 10% | R² |
|---|---|---|---|---|---|
| Gradient boosting | **6.7%** | **5.2%** | **£1,086** | **78.6%** | **0.967** |
| Median for model + year | 13.4% | 9.1% | £2,126 | 53.6% | 0.833 |
| Median for model | 28.4% | 17.8% | £4,022 | 30.2% | 0.599 |

Fair range (split conformal on log residuals): ±10.9%, test coverage 80.0% against an 80% target; by brand 76.0%
(Skoda) to 84.1% (Toyota).

By price band (MAPE): under £7.5k 10.0% · £7.5–12.5k 6.5% · £12.5–20k 6.2% · £20–35k 6.2% · £35k+ 7.2%.
By brand: 6.0% (Toyota) to 7.2% (Skoda).

The released model is refit on train + test (87,882 adverts) with the same calibration split; its range is ±10.6%.

**Intended use.** A second opinion on asking prices, and relative comparisons (depreciation, value) between models.

**Not for.** Valuations for finance, insurance or legal purposes; cars outside the 9 brands; electric cars; today's
pounds without adjusting for the market since 2020.

**Known weaknesses.**
* Cars from 2010 or earlier (MAPE about 25% on 257 test adverts) and very cheap cars: condition and history dominate there.
* No trim level, condition, colour, options, service history or location.
* Random split: near-duplicate adverts across train and test may make the score slightly optimistic.
* Prices are 2020 asking prices, not sale prices.
* Resale estimates extrapolate the age/mileage pattern in 2020 data; they don't know about future market shifts.
