# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:40:59 2017

@author: sergiarmengol
"""

import pandas as pd
import numpy as np
from sklearn.linear_model.base import LinearRegression
import matplotlib.pyplot as plt
import math
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv',dtype = dtype_dict)
sales = sales.sort(['sqft_living','price'])

train_data = pd.read_csv('wk3_kc_house_train_data.csv',dtype = dtype_dict)
train_data = train_data.sort(['sqft_living','price'])

test_data = pd.read_csv('wk3_kc_house_test_data.csv',dtype = dtype_dict)
test_data = test_data.sort(['sqft_living','price'])

valid_data = pd.read_csv('wk3_kc_house_valid_data.csv',dtype = dtype_dict)
valid_data = valid_data.sort(['sqft_living','price'])


msk = np.random.rand(len(sales)) < 0.9




def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = poly_dataframe['power_1'].apply(lambda x: math.pow(x,power))
    return poly_dataframe



'''
Ex 6. compute the regression weights for predicting sales[‘price’] 
based on the 1 degree polynomial feature ‘sqft_living’. The result should be an intercept and slope
'''
poly1_data = polynomial_dataframe(sales['sqft_living'],1)
poly2_data = polynomial_dataframe(sales['sqft_living'],2)
poly3_data = polynomial_dataframe(sales['sqft_living'],3)
poly15_data = polynomial_dataframe(sales['sqft_living'],15)
output = sales['price']

model1 = LinearRegression()
model1.fit(poly1_data,output)

model2 = LinearRegression()
model2.fit(poly2_data,output)
model3 = LinearRegression()
model3.fit(poly3_data,output)

model15 = LinearRegression()
model15.fit(poly15_data,output)

print model1.intercept_
print model1.coef_
    
plt.plot(poly1_data['power_1'],output,'.',
poly1_data['power_1'], model1.predict(poly1_data),'-')
plt.plot(poly2_data['power_1'],output,'.',
poly2_data['power_1'],model2.predict(poly2_data),'-')
plt.plot(poly3_data['power_1'],output,'.',
poly3_data['power_1'],model3.predict(poly3_data),'-')
plt.plot(poly15_data['power_1'],output,'.',
poly15_data['power_1'],model15.predict(poly15_data),'-')

'''
Estimate a 15th degree polynomial on all 4 sets, plot the results and view the coefficients for all four models.
'''

# TRAIN DATA
train15_data = polynomial_dataframe(train_data['sqft_living'],15)
train_model15 = LinearRegression()
train_model15.fit(train15_data,train_data['price'])

plt.plot(train15_data['power_1'],train_data['price'],'.',
train15_data['power_1'], train_model15.predict(train15_data),'-')

# VALID DATA
valid15_data = polynomial_dataframe(valid_data['sqft_living'],15)
valid_model15 = LinearRegression()
valid_model15.fit(valid15_data,valid_data['price'])

plt.plot(valid15_data['power_1'],valid_data['price'],'.',
valid15_data['power_1'], valid_model15.predict(valid15_data),'-')

# TEST DATA
test15_data = polynomial_dataframe(test_data['sqft_living'],15)
test_model15 = LinearRegression()
test_model15.fit(test15_data,test_data['price'])

plt.plot(test15_data['power_1'],test_data['price'],'.',
test15_data['power_1'], test_model15.predict(test15_data),'-')

'''
ex 15. Now for each degree from 1 to 15:

Build an polynomial data set using training_data[‘sqft_living’] as the feature and the current degree
Add training_data[‘price’] as a column to your polynomial data set
Learn a model on TRAINING data to predict ‘price’ based on your polynomial data set at the current degree
Compute the RSS on VALIDATION for the current model (print or save the RSS)
'''
rss = np.array([])

for i in range(1,15+1) :
    train_poly_data = polynomial_dataframe(train_data['sqft_living'],i)
    train_poly_data['price'] = train_data['price']
    train_poly_model = LinearRegression()
    train_poly_model.fit(train_poly_data,train_data['price'])
    
    valid_poly_data = polynomial_dataframe(valid_data['sqft_living'],i)
    predictions = train_poly_model.predict(valid_poly_data)
    residual = valid_data['price'] - predictions
    rss = np.append(rss,sum(residual*residual))


'''
ex 16. Quiz Question: Which degree (1, 2, …, 15) had the lowest RSS on Validation data?
'''
print np.argmin(rss)
print min(rss)
 
'''
Compute the RSS on TEST data for the model with the best degree from the Validation data.
'''
 
train_poly_data = polynomial_dataframe(train_data['sqft_living'],6)
#train_poly_data['price'] = train_data['price']
train_poly_model = LinearRegression()
train_poly_model.fit(train_poly_data,train_data['price'])

test_poly_data = polynomial_dataframe(test_data['sqft_living'],6)
predictions = train_poly_model.predict(test_poly_data)

residual = test_data['price'] - predictions
rss = sum(residual*residual)
print rss
        


