# Banking-Finance-Project
## Objective

To develop a real-time predictive fraud detection model for PredCatch Analytics’ Australian banking client that identifies and prevents fraudulent ATM transactions. The goal is to reduce, and ideally eliminate, fraud-related financial losses and reputational damage by accurately detecting fraudulent transactions despite a highly imbalanced dataset. If successful, the solution will be presented to the client with a clear explanation of its working and impact.

## Target

The prediction target is the variable Target, which indicates the transaction class:

1 → Fraudulent transaction

0 → Clean (legitimate) transaction

The model’s task is to correctly classify each transaction as fraudulent or non-fraudulent based on the provided masked transaction features and additional engineered risk indicators (geo_scores, Lambda_wts, Qset_tats, instance_scores).
