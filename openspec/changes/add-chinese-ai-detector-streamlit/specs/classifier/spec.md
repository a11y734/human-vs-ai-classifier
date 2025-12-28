## ADDED Requirements

### Requirement: Chinese + English AI-vs-Human Classification
The system MUST train and use a baseline classifier to label Chinese and English text as AI-generated or Human-written and MUST produce probability outputs for both classes.

#### Scenario: Train baseline model from HC3-Chinese directory
- WHEN the user runs the training routine on a local HC3-Chinese dataset directory containing `.csv` files
- THEN the system loads the HC3-Chinese data (fields `question`, `human_answers`, `chatgpt_answers`) and flattens answers into labeled samples
- AND labels human answers as Human/0 and ChatGPT answers as AI/1
- AND performs a stratified train/validation split with a fixed random seed
- AND trains a TF‑IDF + Logistic Regression/SVM model
- AND saves artifacts (vectorizer and model) under `models/` in a versioned run directory
- AND outputs evaluation metrics (accuracy, F1, ROC‑AUC, PR‑AUC) and plots under the run's `metrics/` folder
- AND records dataset stats (class counts, text length distribution)

#### Scenario: Training without reference dataset
- GIVEN the user disables the reference CSV
- WHEN the training routine runs
- THEN the system trains using only the HC3-Chinese inputs

#### Scenario: Bilingual training option
- GIVEN the bilingual flag `--include-english` is enabled and an English reference dataset exists under `refs/human-vs-ai-text-classifier/data/*.csv` with columns (`text`, `label`)
- WHEN the training routine runs
- THEN the system merges the English and Chinese datasets, normalizes schemas and labels, and performs a stratified split across classes and languages with a fixed random seed
- AND applies class/language balancing (via sampling or class weights) to avoid skew toward a single language or class

#### Scenario: Chinese script unification
- WHEN training or inference receives Traditional Chinese text
- THEN the system treats it as Chinese (same language category as Simplified)
- AND SHOULD normalize to Simplified during preprocessing for feature consistency; if normalization is unavailable, character n-gram features MUST be used to mitigate script variance

#### Scenario: English input inference (bilingual artifacts)
- GIVEN bilingual artifacts exist from a run with `--include-english`
- WHEN the user provides an English text to the inference module
- THEN the system returns probabilities for both classes (AI%, Human%) and the predicted label with confidence

#### Scenario: Predict single input
- GIVEN trained artifacts exist under `models/`
- WHEN the user provides a Chinese text to the inference module
- THEN the system returns probabilities for both classes (AI%, Human%)
- AND returns the predicted label with confidence

### Requirement: Streamlit UI for Inference
The application MUST provide a Streamlit UI that accepts a single text input, MUST display prediction probabilities, the predicted label, and confidence, MUST use Chinese labels for user-facing text, and MUST present a styled layout with clear visual hierarchy and high-contrast blocks. It MUST present evaluation metrics and dataset stats in a readable format and SHOULD render evaluation visuals when available.

#### Scenario: Interactive prediction
- WHEN the user pastes text and clicks predict
- THEN the UI immediately shows AI% and Human% with the predicted label and confidence

#### Scenario: Visualizations and stats
- WHEN evaluation artifacts exist
- THEN the UI can display confusion matrix and ROC/PR curve images and dataset stats

#### Scenario: Low-confidence indication
- WHEN the AI and Human probabilities are close
- THEN the UI indicates the result is low-confidence or uncertain and suggests providing more text

#### Scenario: Chinese UI localization
- WHEN the user opens the UI
- THEN headings, labels, and buttons are presented in Chinese with a polished, high-contrast layout

#### Scenario: Missing artifacts handling
- GIVEN no trained artifacts are present
- WHEN the user opens the UI
- THEN the UI clearly indicates training is required and points to the training command

### Requirement: Optional Explanation
The system SHALL provide an optional explanation or feature-importance summary for baseline predictions when supported by the model artifacts.

#### Scenario: Explanation available
- GIVEN baseline artifacts include feature weights
- WHEN the user views the prediction in the UI
- THEN the system shows a short list of influential features or a summary, or indicates that explanation is unavailable
