# Predictive Fraud Detection for ATM Transactions

## Objective

Develop a real-time predictive fraud detection model for PredCatch Analytics' Australian banking client that identifies fraudulent ATM transactions despite a highly imbalanced dataset, reducing fraud-related financial losses and reputational damage. The target variable is `Target` (1 = fraudulent transaction, 0 = legitimate transaction), predicted from masked transaction features plus engineered risk indicators.

## Dataset

- **Core data:** [`train.csv`](train.csv) (227,845 transactions) and [`test_share.csv`](test_share.csv) (56,962 unseen transactions), 27 raw columns — behavioral percentiles (`Per1`–`Per9`), demographic indicators (`Dem1`–`Dem9`), credit indicators (`Cred1`–`Cred6`), and `Normalised_FNT`.
- **Engineered risk features**, merged in on `id`/`Group`:
  - [`Geo_scores.csv`](Geo_scores.csv) — `geo_score` per transaction
  - [`Qset_tats.csv`](Qset_tats.csv) — `qsets_normalized_tat` per transaction
  - [`instance_scores.csv`](instance_scores.csv) — `instance_scores` per transaction
  - [`Lambda_wts.csv`](Lambda_wts.csv) — `lambda_wt` per account `Group` (1,400 groups)
- After merging, each transaction carries 33 features.
- **Severe class imbalance:** only 394 of 227,845 training transactions (0.17%) are fraudulent.

## Key Insights

- Fraud is extremely rare (~0.17% of transactions), so raw accuracy is meaningless here — precision and recall on the fraud class are what actually matter.
- A Random Forest classifier trained on the merged feature set reached **96% precision and 68% recall (F1 = 0.80)** on held-out fraudulent transactions: when it flags a transaction as fraud it's right 96% of the time, and it catches roughly two-thirds of actual fraud cases.
- On the fully unseen test set (56,962 transactions), the model flagged **82 transactions as fraudulent (~0.14%)** — consistent with the training-set fraud rate, suggesting the model generalizes without over- or under-flagging.
- The engineered risk indicators (`geo_score`, `qsets_normalized_tat`, `instance_scores`, `lambda_wt`) were essential additions — the raw `Per`/`Dem`/`Cred` features alone aren't enough to reliably separate fraud at this level of class imbalance.
- At 68% recall, roughly 1 in 3 fraud cases still slips through — the model currently favors precision (few false alarms) over catching every fraud case, a trade-off worth revisiting with techniques like SMOTE or class-weighting if recall is the priority.

## Tech Stack

Python · Pandas · scikit-learn (Random Forest)
