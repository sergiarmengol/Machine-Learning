# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:52:02 2017

@author: sergiarmengol
"""


import pandas as pd
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
import pprint

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

def polynomial_dataframe(feature, degree): 
    '''
    This function accepts an array ‘feature’ (of type pandas.Series) 
    and a maximal ‘degree’ and returns an data frame 
    (of type pandas.DataFrame) with the first column equal to ‘feature’ 
    and the remaining columns equal to ‘feature’ to increasing 
    integer powers up to ‘degree’.
    @params:
            # feature is pandas.Series type
            # assume that degree >= 1 - integer
    '''
    
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
            poly_dataframe[name] = feature.apply(lambda x: x**power)
            
    return poly_dataframe


def ridge_polynomial_regression(csv,l2_penalty,deg,counter) :
    '''
    Fit a degree polynomial on the set from the csv data, 
    plot the results and view the weights for the four models
    @params:
        # csv is a string with the __DIR__ of the csv file to read
        # l2_penalty is  amount of regularization 
        applied on the ridge regression
    '''
    
    # Prepare data to fit the model
    subset = pd.read_csv(csv, dtype=dtype_dict)
    subset = subset.sort(['sqft_living','price']) 
    poly_subset_deg = polynomial_dataframe(subset['sqft_living'],deg)
    my_features = poly_subset_deg.columns
    poly_subset_deg['price'] = subset['price']
    
    # Fit the model with a  degree polynomial
    model = linear_model.Ridge(alpha=-l2_penalty,normalize=True)
    model.fit(poly_subset_deg[my_features],poly_subset_deg['price'])
    print pd.Series(model.coef_,index=my_features) #use print vs. return since return can only apply to function

    plt.plot(poly_subset_deg['power_1'].reshape(len(poly_subset_deg['power_1']),1),poly_subset_deg['price'].reshape(len(poly_subset_deg['price']),1),'.',
             poly_subset_deg['power_1'].reshape(len(poly_subset_deg['power_1']),1), model.predict(poly_subset_deg[my_features]),'-')

l2_small_penalty=1e-9 # small amount of regularization
l2_large_penalty=1.23e2 # big amout of regularization

l2_penalty = l2_small_penalty

#csv_sales ='kc_house_data.csv'
#csv_1 = 'wk3_kc_house_set_1_data.csv'
#csv_2 = 'wk3_kc_house_set_2_data.csv'
#csv_3 = 'wk3_kc_house_set_3_data.csv'
#csv_4 = 'wk3_kc_house_set_4_data.csv'

#ridge_polynomial_regression(csv_sales,l2_small_penalty,15,0)

#Ridge regression

for i in range(1,5) :
    ridge_polynomial_regression('wk3_kc_house_set_'+str(i)+'_data.csv',l2_penalty,15,i)
    
    
# Selecting an L2 penalty via k-fold cross-validation
    
'''
K-fold cross-validation:
it involves dividing the training set into k segments of roughtly equal size. 
Similar to the validation set method, 
we measure the validation error with one of the segments designated as the 
validation set.
The major difference is that we repeat the process k times as follows:

Set aside segment 0 as the validation set, and fit a model on rest of data, 
and evalutate it on this validation set
Set aside segment 1 as the validation set, and fit a model on rest of data, 
and evalutate it on this validation set
...
Set aside segment k-1 as the validation set, 
and fit a model on rest of data, and evalutate it on this validation set
After this process, we compute the average of the k validation errors, 
and use it as an estimate of the generalization error. 
Notice that all observations are used for both training and validation, 
as we iterate over segments of data.
'''
def k_fold_cross_validation(k, l2_penalty, data):    
    rss_sum = 0
    n= len(data)
    for i in range(0,k):
        start = (n*i)/k
        end = (n*(i+1)/k-1)
        
        ## k-fild process per blocks
        validation_set = data[start:end+1]
        training_set = data[0:start].append(data[end+1:n])
        
        ## compute input to have 15-polynomial data ( training set and validation set )
        poly_data = polynomial_dataframe(training_set['sqft_living'],15)
        valid_data = polynomial_dataframe(validation_set['sqft_living'],15)
        poly_data['price'] = training_set['price']
        valid_data['price'] = validation_set['price']
        poly_features = poly_data.columns
        valid_features = valid_data.columns
        
        # Fit the model with the training data
        model = linear_model.Ridge(alpha=l2_penalty,normalize=True)
        model.fit(poly_data[poly_features],poly_data['price'])
        
        ## Compue rss of the model
        prediction = model.predict(valid_data[valid_features])
        residual = valid_data['price']-prediction
        rss = np.sum(residual*residual)
        rss_sum += rss
        
    validation_error = rss_sum/k
    
    return validation_error


# Train data is 90%

'''
Get the l2_penalty that minimize the RSS of the model 
with the training  data with a 10-fild-cross-validation
'''
train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
k = 10 # 10-fold cross-validation
val_error_dict = {}
for l2_penalty in np.logspace(3, 9, num=13) :
    val_err = k_fold_cross_validation(k, l2_penalty, train_valid_shuffled)   
    val_error_dict[l2_penalty] = val_err

#pprint.pprint(val_error_dict)

'''
Using the best L2 penalty found above, 
train a model using all training data. 
What is the RSS on the TEST data of the model you learn with this L2 penalty?
'''
## Test data is 10%
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)

poly_data = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)
test_poly_data = polynomial_dataframe(test['sqft_living'], 15)

test_features = test_poly_data.columns
train_features = poly_data.columns

poly_data['price'] = train_valid_shuffled['price']
test_poly_data['price'] = test['price']


# Build a model using the training data and the best l2_penalty find by k-fold cross-validation
model_test = linear_model.Ridge(alpha=1000.0,normalize=True)
model_test.fit(poly_data[test_features],poly_data['price'])

# Comute test data RSS with the model learned by the training data
prediction = model_test.predict(test_poly_data[test_features])
RSS_TEST = np.sum((prediction-test_poly_data['price'])**2)

print "Rss test: " + str(RSS_TEST)

