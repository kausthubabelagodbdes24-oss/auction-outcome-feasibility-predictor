# Auction Outcome Feasibility

This context describes an academic predictor that approximates formal model-checking results for configurations in the UCI Auction Verification dataset. It does not evaluate real-world auctions for fraud, authenticity, fairness, or legal validity.

## Language

**Auction Configuration**:
The seven bidder-capacity, price, product, and potential-winner values supplied for one modeled verification run.
_Avoid_: Real auction, auction transaction

**In-Domain Configuration**:
An Auction Configuration whose individual values all fall within the published dataset domains.
_Avoid_: Valid auction, verified configuration

**Novel Configuration**:
An In-Domain Configuration whose exact seven-value combination does not occur in the model's training data.
_Avoid_: Invalid configuration, impossible auction

**Model Outcome**:
A price and, when specified, winner assignment evaluated within the modeled simultaneous multi-round auction.
_Avoid_: Real-world auction result, verified transaction

**Outcome Feasibility**:
Whether a Model Outcome is possible within the modeled auction according to the dataset's binary verification result.
_Avoid_: Auction validity, authenticity, fairness, fraud status, legal validity

**Outcome Feasibility Prediction**:
An SVM estimate of Outcome Feasibility used as an academic surrogate for formal model checking.
_Avoid_: Auction verification, fraud detection, real-world validation

**Verification Result**:
The source dataset field `verification.result`, used as the binary label for Outcome Feasibility.
_Avoid_: Verified auction, legal verdict
