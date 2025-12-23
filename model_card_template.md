# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a supervised binary classification model trained to predict whether an individual earns more than $50,000 per year based on U.S. Census data.
The model was developed as part of the Udacity MLOps Nanodegree project *Deploying a Scalable ML Pipeline with FastAPI*.

The model uses a **Random Forest classifier** implemented with scikit-learn.
Categorical features are one-hot encoded, and the target variable is the `salary` column.

The model was trained and evaluated using Python, pandas, scikit-learn, and FastAPI-compatible utilities.

## Intended Use

The intended use of this model is **educational and demonstrative**, showcasing how to build, train, evaluate, and deploy an ML pipeline using best practices.

Primary intended users are students and practitioners learning MLOps concepts.
This model is **not intended for real-world decision-making**, such as hiring, lending, or income verification.

Out-of-scope use cases include any automated decision system affecting individuals’ financial or employment outcomes.

## Training Data

The training data is derived from the U.S. Census Income dataset (`census.csv`).
It includes demographic and employment-related features such as age, workclass, education, occupation, race, sex, hours worked per week, and native country.

The dataset was split into training and testing subsets before model training.
Preprocessing included handling categorical variables via one-hot encoding and separating the label (`salary`) from feature data.

## Evaluation Data

The evaluation data consists of a held-out test split from the same Census dataset.
This dataset was not used during training and was processed using the fitted encoder and label binarizer from the training phase.

Model performance was evaluated both on the full test dataset and on **categorical data slices** to assess performance consistency across subgroups.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model was evaluated using the following metrics:

- **Precision**
- **Recall**
- **F1 Score**

Overall model performance on the test dataset:

- **Precision:** 0.7263
- **Recall:** 0.6472
- **F1 Score:** 0.6845

In addition, performance metrics were computed for slices of the data based on categorical features (e.g., `workclass`).
These slice-based metrics were saved to `slice_output.txt` to help identify disparities in model performance across subgroups.

## Ethical Considerations

This model is trained on historical census data, which may reflect societal biases related to income, race, gender, and occupation.
As a result, predictions made by this model may propagate or amplify existing biases.

The model should **not** be used in real-world systems that impact individuals’ livelihoods or opportunities.
Care must be taken when interpreting performance metrics, especially for underrepresented groups with small sample sizes.

## Caveats and Recommendations

- The dataset contains imbalanced classes, which may affect recall for higher-income predictions.
- Some categorical slices have very small sample sizes, leading to unstable or misleading performance metrics.
- This model has not been optimized for fairness, robustness, or production deployment.

Future improvements could include:
- Hyperparameter tuning
- Fairness-aware evaluation and mitigation
- Cross-validation
- Threshold optimization
- Monitoring performance drift over time

This model is intended strictly for instructional purposes.

