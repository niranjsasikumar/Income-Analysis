import pandas as pd

# Load data from CSV file
data = pd.read_csv('./data/raw_data.csv')

# Verify the columns that have missing values
missing_values_columns = ['workclass', 'occupation', 'native-country']
for column in missing_values_columns:
    print(f'{column} has missing values: {data[column].isnull().any()}')

# Drop rows with missing values
print(data.shape)
data = data.dropna()
print(data.shape)

# Drop rows with '?' in above columns
data = data[~(data == '?').any(axis=1)]
print(data.shape)

# Drop education column
print(len(data['education'].unique()))
print(data['education-num'].min(), data['education-num'].max())
data = data.drop(columns='education')

# Drop fnlwgt column
data = data.drop(columns='fnlwgt')

# Fix income column
income_mapping = { '>50K': 1, '>50K.': 1, '<=50K': 0, '<=50K.': 0 }
data['income'] = data['income'].map(income_mapping)
print(data['income'].unique())

# Change native-country column to us-native column
data['native-country'] = data['native-country'].apply(lambda x: 1 if x == 'United-States' else 0)
data = data.rename(columns={'native-country': 'us-native'})
print(data['us-native'].unique())

print(data.shape)

# Save cleaned data to a local CSV file
data.to_csv('./data/clean_data.csv', index=False)