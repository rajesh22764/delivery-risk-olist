# Model card: late-delivery risk model

## Summary
A regularised logistic regression (29 features) that scores an e-commerce order, **at the moment it is placed**, for the risk of being delivered after its promised date. Trained and evaluated on Olist marketplace orders (Brazil, 2017 to 2018). It is an **offline study**, not a production system.

## Intended use
- **Intended:** ranking orders by risk so that a limited intervention (a faster carrier, a more honest promise, an early customer message) can target the riskiest ones; as a learning example of an end-to-end tabular ML workflow.
- **Not intended:** deciding anything about customers or sellers as individuals; use on other marketplaces, countries or years without re-validation; any use that treats the probabilities as reliable in unusual periods (peaks, strikes, holidays).

## Data
Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0); national holidays (Nager.Date) and daily weather (Open-Meteo), the latter not used by the final model. Training: 46,461 delivered orders purchased 2017-01-01 to 2018-01-14. Validation: 6,802 orders (March 2018). Test: 20,874 orders purchased 2018-05-15 to 2018-08-21. Orders that were never delivered are absent.

## Inputs (known at purchase time only)
Promised delivery date and purchase time; order value, freight and payment; seller and customer location (state, distance); the seller's recent delivery record; recent late rates into the customer's state and platform-wide; holidays before the promised date. Nothing from after purchase (no carrier hand-off time, no approval delay, no realised weather).

## Performance (test window, single evaluation, 95% bootstrap intervals)
See the README for the full tables. ROC-AUC 0.720 [0.705, 0.735]; PR-AUC 0.100 (base rate 4.2%); 24% of late orders fall in the riskiest 10% of orders. A 5-feature model (promise and calendar only) is statistically indistinguishable (ROC-AUC 0.703). Gradient boosting was worse (0.645).

## Known limitations and failure modes
- **Regime dependence.** Late rates swing between about 1% and 19% by month. Raw probabilities were close to calibrated in the test period but badly low in a spike month (predicted 6.8%, actual 18.6%). Do not use absolute probabilities across regimes; use the percentile rank.
- **Decay.** ROC-AUC fell from 0.75 (May to June 2018) to 0.61 (August 2018). Retrain regularly on orders old enough to have resolved (45+ days) and monitor weekly.
- **Modest signal, so limited value.** Acting on flags saves essentially nothing at a late-order cost of 10x an intervention (50% success); about 12% at 40x. The cost figures are assumptions; no intervention was tested.
- **Seller effects are not persistent**, so the model cannot reliably single out "bad sellers".
- **Right-censoring and missing orders.** Lost or undelivered orders, and late deliveries of the newest orders, are not in the data.
- **Distribution shift in inputs** is large (population stability index above 1 for the holiday and platform-lateness features between training and test).

## Fairness and risk considerations
The model uses the customer's and seller's **state** as inputs. If its score were used to decide who receives a longer promise, a slower service or worse treatment, regions that are already harder to serve could be disadvantaged. Before any real use: audit performance and flag rates by region, prefer actions that help (expedite, communicate) over actions that penalise, and keep a human-reviewable explanation (the model is linear, so contributions are exact).

## Maintenance
Retrain monthly or when weekly ROC-AUC drops below 0.60 or any input's population stability index exceeds 0.25. Keep `scikit-learn` pinned to the training version (the model is a pickled pipeline). Statistics used for serving are looked up from tables built as of a snapshot date and must be refreshed with each retrain.
