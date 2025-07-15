# Install necessary packages
# pip install aif360 pandas matplotlib seaborn scikit-learn

from aif360.datasets import CompasDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
dataset = CompasDataset()

# Split into privileged (Caucasian) and unprivileged (African-American) groups
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

# Split dataset
train, test = dataset.split([0.7], shuffle=True)

# Reweighing to mitigate bias
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
train_transf = RW.fit_transform(train)

# Train logistic regression on transformed data
X_train = train_transf.features
y_train = train_transf.labels.ravel()

X_test = test.features
y_test = test.labels.ravel()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create new dataset with predictions
test_pred = test.copy()
test_pred.labels = y_pred

# Calculate fairness metrics
metric = ClassificationMetric(test, test_pred,
                              unprivileged_groups=unprivileged_groups,
                              privileged_groups=privileged_groups)

print("Disparate Impact:", metric.disparate_impact())
print("Equal Opportunity Difference:", metric.equal_opportunity_difference())
print("False Positive Rate Difference:", metric.false_positive_rate_difference())

# Visualization
data = {
    'Metric': ['Disparate Impact', 'Equal Opportunity Diff', 'FPR Diff'],
    'Value': [metric.disparate_impact(), 
              metric.equal_opportunity_difference(), 
              metric.false_positive_rate_difference()]
}

df = pd.DataFrame(data)
sns.barplot(x='Metric', y='Value', data=df)
plt.title("Bias Metrics for COMPAS Dataset")
plt.ylim(-1, 2)
plt.axhline(1, color='grey', linestyle='--')
plt.axhline(0, color='red', linestyle='--')
plt.show()
