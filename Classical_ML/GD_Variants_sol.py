'''
In this problem, you need to implement a single function that can perform three variants of gradient descent: Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent using Mean Squared Error (MSE) as the loss function. The function will take an additional parameter to specify which variant to use.

Requirements
Do not shuffle the data; process samples in their original order (index 0, 1, 2, ...)
For Batch GD: use all samples to compute a single gradient update per epoch
For Stochastic GD: iterate through each sample sequentially (i.e., process sample 0, then 1, then 2, etc.) â not randomly selected
For Mini-Batch GD: form batches from consecutive samples without overlap (e.g., for batch_size=2: first batch uses indices [0,1], second batch uses [2,3], etc.)
The n_epochs parameter specifies how many complete passes through the dataset to perform
For each epoch, process all samples according to the specified method
'''


import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape
    
    for epoch in range(n_epochs):
        if method == 'batch':
            y_hat = X @ weights #(m,n) * (n) = (m,1)
            error = y - y_hat #(m,1)
            grad = (-2/m) * (X.T @ error) #(n,m) & (m,1) = (n,1)

            weights = weights - learning_rate * grad

        elif method == 'stochastic':
            # process each sample at a time
            for i in range(m):
                xi = X[i, :].reshape(1, -1) # (1, n)
                #print("xi.shape: ", xi.shape)
                yi = y[i] # (1)
                #print("yi : ", yi)

                y_hat = xi @ weights # (1,n) * n = 1
                #print("y_hat: ", y_hat.shape)

                error = yi - y_hat # 1
                #print("error shape: ", error.shape)

                grad = -2 * (xi.T) * error # (n,1) * 1 = (n,1)
                grad = grad.flatten() #(n,)
                #print(grad.shape)

                #print("weight shape: ", weights.shape)

                weights = weights - learning_rate * grad
        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size] #(b,n)
                y_batch = y[i:i+batch_size] #(b)

                y_hat = X_batch @ weights #(b,n) * (n,) = (b,)

                error = y_batch - y_hat #(b,)

                grad = (-2/batch_size) * (X_batch.T @ error) # (n,b) * (b,) = (n,)

                weights = weights - learning_rate * grad # (n,)
    

    return weights


