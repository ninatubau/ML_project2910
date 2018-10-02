##
import numpy as np

#HELPER FUNCTIONS
def load_data_project(sub_sample = True):
    """Load data and convert it to the metrics system."""
    path_dataset = "training.csv"
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=range(30))
    labels = np.genfromtxt(
        path_dataset, dtype='str', delimiter=",", skip_header=1, usecols=[32])

    # sub-sample
    if sub_sample:
        data = data[::100]
        labels = labels[::100]
        
    return data, labels

def standardize_columns(data, account_for_missing = True):
    """Standardize the original data set."""
        #account_for_missing indicates whether the average value & standard deviation for a parameter should be calculated included values        #-999 (which it obviously shouldn't, as -999 only indicates whether the value is present) or not. If it is true, after standarization missing values are set to 0, so as to not affect the output
    x = np.zeros(data.shape)
    if (account_for_missing): 
        print('We account for missing values')
        missing_values = np.zeros(data.shape,dtype = bool)
        mean_x= np.zeros(data.shape[1]) #value in position i is the average value of parameter i 
        std_x= np.zeros(data.shape[1]) #value in position i is the standard deviation of parameter i 
        for i in range(data.shape[1]):
            missing_values[data[:,i]==-999,i] = 1
            mean_x[i] = np.mean(data[missing_values[:,i]!=1,i])
            std_x[i] = np.std(data[missing_values[:,i]!=1,i])
            x[:,i] = (data[:,i] - mean_x[i]) / std_x[i]
        x[missing_values] = 0
       
    else: 
        missing_values = 0 
        mean_x = np.mean(data,0)
        x = data - mean_x
        std_x = np.std(x,0)
        x = x / std_x
    return x, mean_x, std_x, missing_values

#I ADDED 1 TO THE NAME TO DIFFERENTIATE FROM THE FUNCTION WE USED IN LAB 2
def build_model_data1(data, labels, missing_values, input_missing = False):
    """Form (y,tX) to get regression data in matrix form."""
    #variable input_missing indicates whether the positions of the missing values should be included as input
    y = labels
    if(input_missing): 
        tx = np.concatenate((np.ones((data.shape[0],1)), data, missing_values),1)
        #tx = np.c_[np.ones(data.shape[0]), data, input_missing]
    else: 
        tx = np.c_[np.ones(data.shape[0]), data]
    return y, tx


def compute_loss(y, tx, w):
    """Calculate the loss using MSE
    """
    f_x = tx.dot(w)
    return sum(pow(y-f_x,2))

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    grad_w=(y-tx.dot(w)).dot(tx)*(-1/len(y))
    return grad_w

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad_w,grad_b = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - np.multiply(grad_w, gamma)
        # TODO: update w by gradient
        # ***************************************************
        # store w and loss
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w



#copy pasted from what the professor gave us in lab2
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


            
#MODEL FUNCTIONS (THEY ALL RETURN LOSS, W)
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad_w = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - np.multiply(grad_w, gamma)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w

#this one is still not finished
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    batch_size = 1 #ideally this would be an input to the function
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y,tx,batch_size):
                gradients = compute_gradient(minibatch_y, minibatch_tx, w)
                w = w - np.multiply(gamma, gradients)
                loss = compute_loss(minibatch_y,minibatch_tx,w)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
        #compute real loss for all samples
        loss = compute_loss(y,tx,w)
    return loss, w

def least_squares(y, tx): 
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = compute_loss(y,tx,w)
    return loss, w

