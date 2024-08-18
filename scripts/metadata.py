import pandas as pd

# Load data from CSV file
data = pd.read_csv('./data/data.csv')
print(data.head())

# Display metadata

print(data.shape)
print(data.dtypes)

numerical = data.select_dtypes(include='number').columns
for column in numerical:
    print(f'{column}\tmin: {data[column].min()}\tmax: {data[column].max()}')

categorical = data.select_dtypes(exclude='number').columns
for column in categorical:
    print(f'{column}:', data[column].unique())