import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load data from CSV file
data = pd.read_csv('./data/clean_data2.csv')
print(data.head())

# One-hot encoding for categorical features

data = data.reset_index(drop=True) # Reset indices since rows were dropped

features_to_encode = ['workclass', 'marital-status', 'occupation', 'race', 'sex']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(data[features_to_encode])
encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
data = data.join(encoded_data).drop(features_to_encode, axis=1)

print(data.shape)

# Split dataset into training and testing sets
X = data.drop(columns='income')
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Fit logistic regression classifiers to tune hyperparameters

l1_acc = []
l2_acc = []
log_range = np.logspace(-5, 5, num=11, base=2)

for c in log_range:
    lrc = LogisticRegression(solver='liblinear', penalty='l1', C=c, random_state=1234).fit(X_train, y_train)
    y_pred = lrc.predict(X_test)
    l1_acc.append(accuracy_score(y_test, y_pred))
    
    lrc = LogisticRegression(solver='liblinear', penalty='l2', C=c, random_state=1234).fit(X_train, y_train)
    y_pred = lrc.predict(X_test)
    l2_acc.append(accuracy_score(y_test, y_pred))

print('L1 best accuracy:', np.max(l1_acc))
print('L1 best C:', log_range[np.argmax(l1_acc)])

print('L2 best accuracy:', np.max(l2_acc))
print('L2 best C:', log_range[np.argmax(l2_acc)])

# Plot accuracies for varying hyperparameters of logistic regression classifier

plt.plot(log_range, l1_acc, label='l1')
plt.plot(log_range, l2_acc, label='l2')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C for L1 and L2 Regularization')
plt.legend()
plt.show()

# Fit initial decision tree classifier to get max_depth

dt = DecisionTreeClassifier(random_state=1234).fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Tree depth:', dt.get_depth())

# Fit decision tree classifiers to tune hyperparameters

dt_acc = []
for d in range(1, 48):
    dt = DecisionTreeClassifier(max_depth=d, random_state=1234).fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dt_acc.append(accuracy_score(y_test, y_pred))

print('Best accuracy:', np.max(dt_acc))
print('Best maximum depth:', np.argmax(dt_acc) + 1)

# Plot accuracies for varying hyperparameters of decision tree classifier

plt.plot(dt_acc)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth')
plt.show()

# Fit random forest classifiers to tune hyperparameters

rfc_acc = []
forest_sizes = [50, 100, 150, 200]
max_depths = [10, 20, 30, 40, 50]
best = (0, 0, 0.0) # (k, d, acc)

for k in forest_sizes:
    k_acc = []
    for d in max_depths:
        rfc = RandomForestClassifier(n_estimators=k, max_depth=d, random_state=1234).fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > best[2]:
            best = (k, d, acc)
        k_acc.append(acc)
    rfc_acc.append(k_acc)

print('Best accuracy:', best[2])
print('Best number of trees:', best[0])
print('Best maximum depth:', best[1])

# Plot accuracies for varying hyperparameters of random forest classifier

for i in range(len(forest_sizes)):
    plt.plot(max_depths, rfc_acc[i], label=f'k = {forest_sizes[i]}')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth for Varying Forest Sizes')
plt.legend()
plt.show()

# Fit AdaBoost classifier and find best boosting iteration

abc = AdaBoostClassifier(n_estimators=200, algorithm='SAMME', random_state=1234)
abc = abc.fit(X_train.values, y_train.values)
abc_acc = [acc for acc in abc.staged_score(X_test.values, y_test.values)]
print('Best accuracy', np.max(abc_acc))
print('Best number of estimators', np.argmax(abc_acc) + 1)

# Plot accuracies for AdaBoost classifier over boosting iterations

plt.plot(abc_acc)
plt.xlabel('Boosting Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Boosting Iterations')
plt.show()