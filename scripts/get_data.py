# Import UC Irvine Machine Learning Repository package
from ucimlrepo import fetch_ucirepo

# Fetch dataset
adult = fetch_ucirepo(id=2)

# Get data as a Pandas dataframe
data = adult.data.original

# Save data to a local CSV file
data.to_csv('./data/data.csv', index=False)