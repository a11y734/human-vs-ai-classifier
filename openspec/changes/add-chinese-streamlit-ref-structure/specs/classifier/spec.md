## ADDED Requirements

### Requirement: Chinese AI-vs-Human Baseline Classifier
The system MUST train a baseline Chinese text classifier from Excel files under `Chinesedata/` and output probability scores for AI and Human classes.

#### Scenario: Train from Excel folder
- WHEN the user runs the training command targeting `Chinesedata/`
- THEN the system loads all `.xlsx` files, normalizes columns (`text`, `label`), maps labels {Human→0, AI→1}
- AND trains TF-IDF (character n-grams) + Logistic Regression with a stratified split and fixed random seed
- AND saves artifacts to `model/` and metrics/plots to `model/metrics/`

#### Scenario: Predict single input
- GIVEN trained artifacts exist in `model/`
- WHEN the user submits text
- THEN the system returns AI% and Human% and a predicted label

#### Scenario: Missing artifacts handling
- GIVEN model artifacts are missing
- WHEN the user starts the UI
- THEN the UI clearly instructs how to run training before inference

### Requirement: Streamlit UI for Inference
The application MUST provide a Streamlit UI that accepts text input and displays probabilities and the predicted label; it SHOULD render available evaluation visuals.

#### Scenario: Interactive prediction
- WHEN the user enters text and clicks Predict
- THEN the UI shows AI% and Human% and the predicted label

#### Scenario: Visualizations
- WHEN evaluation artifacts exist under `model/metrics/`
- THEN the UI displays confusion matrix and ROC/PR plots or summary stats

