import pandas as pd                             #importing pandas
import matplotlib.pyplot as plt                 #importing matplotlib.pyplot, Each pyplot function makes some change to a figure
from scipy import stats                         #This module contains a large number of probability distributions as well as a growing library of statistical functions.                        
import numpy as np                              #importing numpy

train = pd.read_csv('train.csv')                #reading csv file

# Scatter Plot before removing outlier
garage_area = train['GarageArea']               # taking values of GarageArea into garage_area  
sales_price = train['SalePrice']                # taking values of saleprice into sales_price  
plt.scatter(garage_area, sales_price, alpha=.75, color='b')  # Drawing a scatter plot with garage_area and sales_price in x and y dimensions,Matplotlib allows you to adjust the transparency of a graph plot using the alpha attribute. 
plt.xlabel('Garage Area')                       #labelling x axis
plt.ylabel('Sale Price')                        #labelling y axis
plt.title('Linear Regression Model')            #labelling Title
plt.show()                                      #printing graph

# Removing outlier by using zscore
# Z score is the relationship between MEAN and Standard Deviation.
data_all = pd.concat([train['GarageArea'], train['SalePrice']], axis=1)
z = np.abs(stats.zscore(data_all)) #z score finds out how many standard devations away of my actual value from mean value
threshold = 3                  # I am assumed that the threshold is 3 any row value having us at score greater than 3 would be classfield as outlier
data = data_all[(z < 3).all(axis=1)]
data_anom = data_all[(z >= 3).all(axis=1)]
# Scatter Plot after removing outlier
garage_area = data['GarageArea']
sales_price = data['SalePrice']
plt.scatter(garage_area, sales_price, alpha=.75, color='b')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression Model')
plt.show()