import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv('winequality-red.csv')

# Handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# Top 3 correlation
data_correlation = data.corr(method='pearson')['quality'][:]  #finding correlaton wrt quality
sorted_data = data_correlation.sort_values(kind='quicksort', ascending=False)  # descending order
print("Descending order")
print(sorted_data[0:3])
print("-------------------------")
sorted_data = data_correlation.sort_values(kind='quicksort', ascending=True) #ascending order
print("Ascending order")
print(sorted_data[0:3])
print("-------------------------")

target_data = data.quality         # target column = quality
features_data = data.drop(['quality'], axis=1)    # dropping quality from data

features_train, features_test, target_train, target_test = train_test_split(features_data, target_data, random_state=42, test_size=.33)    # 67 traning and 33 testing

lr = linear_model.LinearRegression()                #fitting into liner regression model
model = lr.fit(features_train, target_train)

# Evaluate the performance

print("R^2 is: \n", model.score(features_test, target_test))            #regression score function defines how your independent and dependent varibales are dependent on each other.
predictions = model.predict(features_test)

print('RMSE is: \n', mean_squared_error(target_test, predictions))      # root mean square error --It represents standard deviation of the differences between predicted values and observed values