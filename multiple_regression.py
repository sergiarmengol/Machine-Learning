# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:13:13 2017

@author: sergiarmengol
"""

import numpy as np
import graphlab as gl
import math
import matplotlib.pyplot as plt


## Data 
sales = gl.SFrame('kc_house_data.gl/')


def get_numpy_data(data_sframe,features,output) :
     ''' Returns 2-D array with all the selected matrix && 1-D array with the output.
    inputs - 
        data_sframe - SFrame with all sales data
        output  - target of our regression
        features - features of our regression
    '''
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    features = ['constant'] + features  ## add new constant value to the array features
    features_sframe = data_sframe[features] # get values of all teh features
    features_matrix = features_sframe.to_numpy() # convert features data to numpy array
    output_sarray = data_sframe[output] # get output data
    output_array = output_sarray.to_numpy() # convert output dats into a numpy array
    return(features_matrix,output_array) # return numpy features and numpy output data

def predict_outcome(feature_matrix,weights) :
    ''' Returns predicted value.
    inputs - 
        feature_matrix - 2-D array of dimensions data points by features
        weights  - 1-D array of estimated regression coefficients
    '''
    return np.dot(feature_matrix,weights)

def feature_derivative(errors,feature):
    ''' Returns jth partial.
    inputs - 
        errors - 1-D array of output-predictions
        feature  - 1-D array of jth position of the features
    '''
    
    partial = 2*np.dot(errors,feature)
    return partial

def regression_gradient_descent(H,y,initial_weights,step_size,tolerance) :
    ''' Returns coefficients for multiple linear regression.
    
    inputs - 
        H - 2-D array of dimensions data points by features
        y - 1-D array of true output
        initial_weights - 1-D array of initial coefficients
        step_size - float, the step size eta
        tolerance  - int, tells the program when to terminate
    '''
    
    converged = False
    w = np.array(initial_weights)

    while not converged :
        predictions = predict_outcome(H,w)
        errors = predictions - y
        gradient_sum_squares = 0
        for i in range(len(w)) :
            partial = feature_derivative(errors,H[:,i])
            gradient_sum_squares += partial**2
            w[i] = w[i] - step_size*partial
        
        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance :
            converged = True

    return w


train_data,test_data = sales.random_split(.8,seed=0)
simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights_ex_9 = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)
print "ex 9: "
print simple_weights_ex_9

(test_simple_feature_matrix,test_output) = get_numpy_data(test_data, simple_features, my_output)
test_predicted_values_ex_11 = predict_outcome(test_simple_feature_matrix,simple_weights_ex_9)
print "ex 11: "
print test_predicted_values_ex_11[0:1]

diff_squared = np.array([(test_output - test_predicted_values_ex_11)**2])
print "RSS ex 12: "
print np.sum(diff_squared)


## second part


model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
simple_weights_ex_13 = regression_gradient_descent(feature_matrix,output,initial_weights, step_size,tolerance)
print "ex 13: "
print simple_weights_ex_13

(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features,my_output)
test_predicted_values_ex_15 = predict_outcome(test_feature_matrix,simple_weights_ex_13)
print "ex 14: "
print test_predicted_values_ex_15[0:1]

diff_squared = np.array([(test_output - test_predicted_values_ex_15)**2])
print "RSS ex 18: "
print np.sum(diff_squared)

print "first house test data"
print test_data['price'][0:1]


