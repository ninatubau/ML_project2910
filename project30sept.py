import numpy as np
import matplotlib.pyplot as plt
import random
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#depending on this 2 variables we use approaches A, C, or none. 
ACCOUNT_FOR_MISSING = True
INPUT_MISSING = True

import datetime
from functionsSep30 import *


data,labels = load_data_project()
labels_int = np.zeros(labels.shape)
labels_int[labels=='s']=1
labels_int[labels=='b']=-1
print(labels_int)

x, mean_x, std_x, missing_values = standardize_columns(data,ACCOUNT_FOR_MISSING)   #if account_for_missing = true, approach A

#works equal or even better slightly better in some cases when considering absurd (-999) values!
#why? --> INFORMATION ABOUT WHICH VALUES ARE MISSING IS IMPORTANT --> ATTEMPT APPROACH c

y, tx = build_model_data1(x, labels_int, missing_values, INPUT_MISSING)   #If input_missing = true the program is approach C 
#(takes as an input also a matrix that has 0 in the present values and 1 in the missing ones). Will give error if  account_for_missing
#is false and input_missing are True. TODO --> solve this so an error is displayed by screen

# Define the parameters of the algorithm.
max_iters = 10
gamma = 0.05

# Initialization
w_initial = np.zeros((tx.shape[1]))

# Find the weigths and the loss using 1 / 6 approaches suggested
start_time = datetime.datetime.now()
#loss, w_final = least_squares_GD(y, tx, w_initial, max_iters, gamma)
#loss, w_final = least_squares_SGD(y, tx, w_initial, max_iters*100, gamma) #to work reasonably well, stochastic gradient descent needs more iterations than GD as we have a very small (1) batch size
#loss, w_final = least_squares(y,tx) #for now, it doesn´t work with approach C (tx has a column that is all zeros and can´t find its inverse)

lambdas=np.logspace(-5,0,15)
for lambda_ in lambdas:
    loss,w_final=ridge_regression(y,tx,lambda_)
    print(lambda_, w_final)

end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()

print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

#check capacity of network to predict output
#score: (we will need to use cross validation in the future, but for now)

predictions = tx.dot(w_final)
predictions[predictions>0] = 1
predictions[predictions<=0]=-1
errors = np.zeros(predictions.shape)
errors[predictions != y] = 1 
nerrors = np.sum(errors)
percentage=nerrors/len(y)
print('you have obtained {n} errors: = {p}%'.format(n=nerrors,p=percentage))



###################################################
#DATA TREATMENT, POSSIBLE APPROACHES
####################################################

#as can be seen, some columns have no missing values, whereas other columns have a lot of missing values
#suggested approach to missing values: 
#without incorporating any treatment of data (account_for_missing = false and input_missing = false): errors = 602

#I measure errors with GD

#Approach A:  --> ALREADY DONE. (account_for_missing = True in standardize_columns function)
#Performanc (error rate measured with GD) = 598 errors
    #1) to calculate the mean and the std in standarization, use only real values (exclude -999)
    #2) set missing values to zero after standarization

#Approach B: --> TODO
        #For each parameter (31 times), the objective is to be able to predict it using the other (available) parameters. 
    #This could be done after standarization. Wparam in this case will be a (31,31) matrix where row i is formed by the parameters
    #used to predict parameter i (other 30 parameters plus bias ). To make an accurate prediction of all parameters
    #use both forward and backward propagation, and combine the results (to deal with entries in which more than 1 parameter
    #is missing)
    
#Apprach C: --> ALREADY DONE (input_missing = true in build_model_data1()) 590 errors
    #what if there is information on WHICH parameters are missing? low performance of approach A makes us think there is. 
    #we could keep them as 0, but add another input which indicates the missing parameters
    #input: concatenation ([p1, p2, ..., pN],[1,0,0,0,1,0,0,...], where a 1 indicates the parameter is missing.
    #Problem: can't use this approach and least squares as one of the intermidiate matrixes doesn´t have an inverse (as some columns
    #are all 0. But if the zeros, are set to -1, the loss diverges... this would need to be solved as we want to compare the performance of 
    #the possible machine learning functions using the different data treatment.
    #TODO --> solve this limitation
    
#Approach D: Combination of B and C --> TODO

#Approach E: Simply exlude of training samples when a parameter is missing (really impractical in this case as for most samples at least
#one parameter is missing) --> TODO
 
#Apprach F: mean of group s and b and add perlin noise (coefficient of standard deviation ) -> to have a bracket
