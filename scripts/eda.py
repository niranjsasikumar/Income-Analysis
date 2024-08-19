import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# Load data from CSV file
data = pd.read_csv('./data/clean_data.csv')
print(data.head())

# Display histogram of numerical features
numerical_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
data.hist(column=numerical_features, figsize=(10,15))
plt.show()

# Analyze capital-gain feature
max_capital_gain = data['capital-gain'].max()
print(max_capital_gain)
print(data[data['capital-gain'] == max_capital_gain].shape[0])

# Drop capital-gain feature
data = data[~(data['capital-gain'] == max_capital_gain)]
print(data.shape)

# Numerical features correlation
print(data[numerical_features].corr())

# Categorical features correlation

def cramers_v(x, y):
    crosstab = pd.crosstab(x, y)
    chi2 = chi2_contingency(crosstab)[0]
    n = crosstab.sum().sum()
    k = min(crosstab.shape) - 1
    return np.sqrt(chi2 / n / k)

categorical_features = [c for c in data.columns if c not in numerical_features]
results = pd.DataFrame(index=categorical_features, columns=categorical_features)
for f1 in categorical_features:
    for f2 in categorical_features:
        results.loc[f1, f2] = round(cramers_v(data[f1], data[f2]), 3)

print(results)

# Remove relationship feature
data = data.drop(columns='relationship')
categorical_features.remove('relationship')

# Display distributions for categorical features

rows = 4
cols = 2
fig, axs = plt.subplots(rows, cols)
fig.delaxes(axs[3][1])
fig.set_figwidth(10)
fig.set_figheight(20)
fig.subplots_adjust(hspace=1.0)

for i in range(rows):
    for j in range(cols):
        if i*2+j == len(categorical_features):
            break
        data[categorical_features[i*2+j]].value_counts().plot(kind='bar', ax=axs[i][j])

plt.show()

print(data.head())

# Save cleaned data to a local CSV file
data.to_csv('./data/clean_data2.csv', index=False)