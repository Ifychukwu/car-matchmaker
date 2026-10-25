# Data

**Kaggle: "100,000 UK Used Car Data set"** (Aditya Desai, scraped from UK used-car adverts in 2020)
https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

1. Click **Download** on the page above (a zip of about 2 MB; free Kaggle account needed).
2. Put the zip, unchanged, in `data/raw/` (any file name ending `.zip`). Or unzip the brand CSVs into `data/raw/`.
3. `python -m carmatch.price_model` trains and evaluates; `streamlit run app.py` starts the app.

Files used: `audi, bmw, ford, hyundi, merc, skoda, toyota, vauxhall, vw.csv` with columns
`model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize` (`hyundi.csv` names the tax column `tax(£)`).
`cclass.csv` and `focus.csv` repeat rows from `merc` and `ford` with fewer columns, and the `unclean` files are the raw
scrape, so they are skipped.

`data/raw/` is git-ignored. Check the dataset's licence on Kaggle before committing the zip (for example, to deploy the
app on Streamlit Community Cloud).

## What cleaning removes (from `reports/metrics.json`)

| step | rows |
|---|---|
| raw rows in the 9 brand files | 99,187 |
| exact duplicate adverts | 1,475 |
| implausible values: year after 2020 (one says 2060), price £500 or less, mpg over 250 (up to 470.8), year before 1990 | 53 |
| too rare to model: 3 "Electric" rows and 9 "Other" gearboxes | 12 |
| **kept** | **97,647 adverts, 194 models** |

`tax` is the annual road tax (VED) shown on the advert in 2020 and `mpg` is the official figure, not real-world.
