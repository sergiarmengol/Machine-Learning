#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:47:10 2017

@author: sergiarmengol
"""
import pandas as pd
import numpy as np
from math import log, sqrt
from sklearn import linear_model,preprocessing  # using scikit-learn
import matplotlib.pyplot as plt
import math
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv',dtype=dtype_dict)
validating = pd.read_csv('wk3_kc_house_valid_data.csv',dtype=dtype_dict)
testing = pd.read_csv('wk3_kc_house_test_data.csv',dtype=dtype_dict)

'''
This function returns a ‘feature_matrix’ (2D array) consisting 
of first a column of ones followed by columns containing the values of 
the input features in the data set in the same order as the input list. It 
alsos return an ‘output_array’ which is an array of the values of the output 
in the data set (e.g. ‘price’).
'''
def get_numpy_data(data_frame,features,output_name) :
    '''
    @params Dataframe,1D array of features,output name
    @returns 2D array feature_matrix
    @output 2D array & 1D array ( numpy arrays )
    '''
    ## Add a column of ones
    data_frame['constant'] = 1
    ## Add the constant column of ones to the beginning of the features
    features = ['constant'] + features
    ## Get all data of the features selected in order
    feat_dataframe = data_frame[features]
    ## Convert the data of the features selected into a numpy array
    feature_matrix = feat_dataframe.as_matrix()
    ## Get teh data of the selected output eg. price
    output_dataframe = data_frame[output_name]
    ## Convert the selected output to a numpy array
    output_array = output_dataframe.as_matrix()
    
    return [feature_matrix,output_array]
 
'''
This function accepts a 2D array ‘feature_matrix’ and a 1D array ‘weights’ 
and return a 1D array ‘predictions’.
'''
def predict_output(feature_matrix,weights) :
    '''
    @params 2D array, 1D array
    @output 1D array of predictions
    y_pred = H·w
    '''
    
    return np.dot(feature_matrix,weights)


'''
To give equal considerations for all features, 
we need to normalize features as discussed in the lectures: we divide each feature by its 2-norm so that the transformed feature has norm 1.
'''
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix/norms
    return (normalized_features, norms)


def lasso_coordinate_descent_step(i,feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix,weights)
    ro_i =(feature_matrix[:,i] * (output-prediction + weights[i] * feature_matrix[:,i])).sum()
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < (-l1_penalty/2.):
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > (l1_penalty/2.):
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.
        
    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):

    weights = np.copy(initial_weights)
    diff = np.copy(initial_weights)
    converged = False
    
    while not converged :
        for i in range(len(weights)) :

            old_weight_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i,feature_matrix, output, weights, l1_penalty)
            diff[i] = np.abs(old_weight_i-weights[i])
#            print( '  -> old weight: ' + str(old_weight_i) + ', new weight: ' + str(weights[i]))
#            print( '  -> abs change (new - old): ' + str(diff[i]))
        
        max_changed = max(diff)
        if(max_changed < tolerance) :
            converged = True
        
        
    return weights

def compute_rss(normalized_feature_matrix,weights,output) :
    prediction =  predict_output(normalized_feature_matrix, weights)
    RSS = np.dot(output-prediction, output-prediction)
    print('RSS for normalized dataset = ' + str(RSS))


def nnz_features(weights,features) :
    feature_list = ['constant'] + features
    feature_weights = dict(zip(feature_list, weights))
    print(feature_weights)
    for k in feature_weights:
        if feature_weights[k] != 0.0 :
            print("")
            print(str(k))


#-----------------------------PART 0------------------------------
            
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)

simple_feature_matrix, norms = normalize_features(simple_feature_matrix)

weights = np.array([1., 4., 1.])

prediction = predict_output(simple_feature_matrix, weights)

ro = np.empty(3)
for i in range(len(weights)):   
    ro[i] =(simple_feature_matrix[:,i] * (output-prediction + weights[i] * simple_feature_matrix[:,i])).sum()

# To find lambda that sets w to 0 -lambda/2 <= ro <= lambda/2 ->
lamb_1 = 2*ro[1]   
lamb_2 = 2*ro[2]   

print(-lamb_2/2<1.4e8<lamb_1/2)
 
#-----------------------------PART 1------------------------------       
#Variables
my_features = ['sqft_living','bedrooms']
my_output = 'price'
initial_weights = np.array([0.,0.,0.])
l1_penalty = 1e7
tolerance = 1.0
#Compute data
feature_matrix,output = get_numpy_data(sales,my_features,my_output)
normalized_features,norms = normalize_features(feature_matrix)
#Calculate weights and RSS
weights = lasso_cyclical_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
compute_rss(normalized_features,weights,output)
nnz_features(weights,my_features)


#-----------------------------PART 2------------------------------    

#Data variables
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']
my_output = 'price'
#Compute data
(feature_matrix, output) = get_numpy_data(training, all_features, my_output)
normalized_feature_matrix,norms = normalize_features(feature_matrix)  
initial_weights = np.zeros(len(all_features)+1)

#Calculate weights and RSS depending on l1_penalty

l1_penalty = 1e7
tolerance=1.0
weights_1e7 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)
compute_rss(normalized_feature_matrix,weights_1e7,output)
nnz_features(weights_1e7,all_features)
weights_normalized1e7 =  weights_1e7 / norms

l1_penalty = 1e8
tolerance=1.0
weights_1e8 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)
compute_rss(normalized_feature_matrix,weights_1e8,output)
nnz_features(weights_1e8,all_features)
weights_normalized1e8 =  weights_1e8 / norms

l1_penalty = 1e4
tolerance=5e5
weights_1e4 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)
compute_rss(normalized_feature_matrix,weights_1e4,output)
nnz_features(weights_1e4,all_features)
weights_normalized1e4 =  weights_1e4 / norms

#-----------------------------PART 3------------------------------    

(test_feature_matrix, test_output) = get_numpy_data(testing, all_features, 'price')

prediction =  predict_output(test_feature_matrix, weights_normalized1e7)
RSS = np.dot(test_output-prediction, test_output-prediction)
print("")
print ('RSS for model with weights1e7 = '+ str(RSS))


prediction =  predict_output(test_feature_matrix, weights_normalized1e8)
RSS = np.dot(test_output-prediction, test_output-prediction)
print("")
print ('RSS for model with weights1e7 = '+ str(RSS))


prediction =  predict_output(test_feature_matrix, weights_normalized1e4)
RSS = np.dot(test_output-prediction, test_output-prediction)
print("")
print ('RSS for model with weights1e4 = '+ str(RSS))


    

