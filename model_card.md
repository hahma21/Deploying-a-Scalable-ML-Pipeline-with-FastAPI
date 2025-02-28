# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model Type: Random Forest Classifier
Libraries Used: sklearn.ensemble.RandomForestClassifier
Hyperparameters:

- Number of estimators: 100
- Random State: 42

## Intended Use

This model is designed to predict whether an individual's income exceeds $50K per year based on census data. It can be used for demographic analysis, policy research, and other socio-economic studies. However, it should not be used for high-stakes decision-making without further validation.

## Training Data

Dataset: The model was trained on the Census Income Dataset.
Features:

- Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Continuous: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
  Preprocessing:
- Categorical features were one-hot encoded.
- Label was binarized (<=50K → 0, >50K → 1).

## Evaluation Data

The dataset was split into 80% training and 20% test sets. The test dataset was processed similarly to the training dataset but used pre-trained encoders.

## Metrics

The model was evaluated using the following classification metrics:

- Precision: 0.7419
- Recall: 0.6384
- F1-score: 0.6863

## Ethical Considerations

Bias & Fairness: The model may exhibit biases present in the census dataset, particularly regarding race, gender, and socio-economic status.

Data Privacy: The dataset contains sensitive demographic attributes. It is essential to ensure proper handling and anonymization.

Limitations: The model does not account for real-world economic changes and may not generalize well beyond the dataset’s time period.

## Caveats and Recommendations

Data Shift: If used on new data, it is recommended to retrain the model periodically to maintain accuracy.

Interpretability: Feature importance analysis should be conducted to ensure fair and explainable decision-making.

Further Improvements: Consider hyperparameter tuning, feature engineering, or using more advanced models like Gradient Boosting or Neural Networks for improved performance.
