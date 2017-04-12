# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:16:27 2017

@author: sergiarmengol

1. Convert a DataFrame into a Numpy array (if applicable)

2. Write a Numpy function to compute the derivative of the regression weights 
with respect to a single feature

3. Write gradient descent function to compute the regression weights given 
an initial weight vector, step size, tolerance, and L2 penalty
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

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
Regression Cost Function: 
Cost(w) = SUM[(prediction-output)^2] + l2_penalty*(w[0]^2+w[1]^2+...+w[k]^2)
Step 1: Unregularized case (RSS): 2*SUM[error*[feature_o]]
Step 2: Regularization term: 2*l2_penalty*w[i] (without the 2*l2_penalty*w[0] term)
''' 
def feature_derivative_ridge(error,feature,weight,l2_penalty,feature_is_constant) :
    unreg_case = 0
    reg_case = 0
    
    unreg_case = 2*np.dot(error,feature)
    
    if(feature_is_constant == False):
        reg_case = 2*l2_penalty*weight
    
    derivative = unreg_case + reg_case
    
    return derivative

    
def ridge_regression_gradient_descent(H,output,w0,step_size,l2_penalty,max_iterations=100) :

    w = np.array(w0) # init the weights
    iterations = 0 # init the iterations
    
    while iterations < max_iterations:
        
        predictions = predict_output(H,w) # predict output based on our model weights
        error = predictions - output # prediction errors
        for i in xrange(len(w)) :     
            # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                feature_is_constant = True
            else :
                feature_is_constant = False
                
            # Compute the derivative for weight[i]]    
            partial = feature_derivative_ridge(error,H[:,i],w[i],l2_penalty,feature_is_constant)

            # compute the new weight based on the error 
            w[i] = w[i] - step_size*partial
        
        iterations += 1 ## next iteration
       
    return w
   
  
## Load data
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('kc_house_data.csv',dtype=dtype_dict)
training_data = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)
test_data = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)

# Model features
simple_features = ['sqft_living']
# Output to predict
my_output = 'price'

# Convert data to matrix
(simple_feature_matrix, output) = get_numpy_data(training_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)


'''
NO REGULARIZATION l2_penalty = 0.0
Ridge regression algorithm to learn the weights of a simple no regularization model
'''
step_size = 1e-12
l2_penalty = 0.0
max_iterations = 1000
w0 = np.array([0.,0.])

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix,output,w0,step_size,l2_penalty,max_iterations)
print "simple_weights_0_penalty: " +str(simple_weights_0_penalty)
'''
HIGH REGULARIZATION l2_penalty = 1e11
Ridge regression algorithm to learn the weights of a simple high regularization model
'''
l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix,output,w0,step_size,l2_penalty,max_iterations)
print "simple_weights_high_penalty: " +str(simple_weights_high_penalty)

print ''

'''
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')

'''
'''
RSS on the TEST data for the following three sets of weights:

1. The initial weights (all zeros)
2. The weights learned with no regularization
3. The weights learned with high regularization
'''
# RSS with initial weights
test_predicted_values = predict_output(simple_test_feature_matrix,np.array([0.,0.]))
RSS_0 = np.array([(test_output - test_predicted_values)**2])
print "Simple model RSS initial weights w0 = 0: " + str(np.sum(RSS_0))

# RSS with weight learned with no regularization
test_predicted_values = predict_output(simple_test_feature_matrix,simple_weights_0_penalty)
RSS_1 = np.array([(test_output - test_predicted_values)**2])
print "Simple model RSS initial weights w0 low l2_penalty: " + str(np.sum(RSS_1))

# RSS with weight learned with high regularization
test_predicted_values = predict_output(simple_test_feature_matrix,simple_weights_high_penalty)
RSS_2 = np.array([(test_output - test_predicted_values)**2])
print "Simple model RSS initial weights w0 high l2_penaly: " + str(np.sum(RSS_2))
print ''

'''
Model with 2 features: [ ‘sqft_living’, ‘sqft_living_15’]
'''

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(training_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

# NO regularization l2_penalty = 0.0
l2_penalty = 0.0
w0 = np.array([0.,0.,0.])
step_size = 1e-12
max_iterations = 1000
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix,output,w0,step_size,l2_penalty,max_iterations)
print ''
print "multiple_weights_0_penalty: " +str(multiple_weights_0_penalty)



# Regularization with high penalty
l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix,output,w0,step_size,l2_penalty,max_iterations)
print "multiple_weights_high_penalty: " +str(multiple_weights_high_penalty)

'''
plt.plot(feature_matrix,output,'k.',
        feature_matrix,predict_output(feature_matrix, multiple_weights_0_penalty),'b-',
        feature_matrix,predict_output(feature_matrix, multiple_weights_high_penalty),'r-')
'''
print ''
# RSS with initial weights
test_predicted_values = predict_output(test_feature_matrix,np.array([0.,0.,0.]))
RSS_0 = np.array([(test_output - test_predicted_values)**2])
print "Multiple model RSS initial weights w0 = 0: " + str(np.sum(RSS_0))

# RSS with weight learned with no regularization
test_predicted_values = predict_output(test_feature_matrix,multiple_weights_0_penalty)
RSS_1 = np.array([(test_output - test_predicted_values)**2])
print "Multiple model RSS initial weights w0 low l2_penalty: " + str(np.sum(RSS_1))
# RSS with weight learned with high regularization
test_predicted_values = predict_output(test_feature_matrix,multiple_weights_high_penalty)
RSS_2 = np.array([(test_output - test_predicted_values)**2])
print "Multiple model RSS initial weights w0 high l2_penalty: " + str(np.sum(RSS_2))


print ''
# Error for the first house of the test data with no regularization
test_predicted = predict_output(test_feature_matrix,multiple_weights_0_penalty)
RSS = np.array([(test_output - test_predicted)**2])
print "RSS low l2_penalty weights : " + str(np.sum(RSS))
print "original" + str(test_output[0:1])
print "predicted" + str(test_predicted[0:1])
print "error" + str(abs(test_output[0:1]-test_predicted[0:1]))

print ''
# Error for the first house of the test data with high regularization
test_predicted = predict_output(test_feature_matrix,multiple_weights_high_penalty)
RSS = np.array([(test_output - test_predicted)**2])
print "RSS high l2_penalty weights : " + str(np.sum(RSS))
print "original" + str(test_output[0:1])
print "predicted" + str(test_predicted[0:1])
print "error" + str(abs(test_output[0:1]-test_predicted[0:1]))

