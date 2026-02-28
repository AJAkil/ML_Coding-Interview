import numpy as np

def gradient_descent(X, y, w, alpha, epochs, batch_size=1, method='batch'):
    '''
    X (m, n)
    y (m,)
    w (n,)
    '''

    m, n = X.shape

    for epoch in epochs:
        if method == 'batch':
            y_hat = X @ w # (m,n) @ (n,) = (m,)
            error = y_hat - y # (m,)
            grad = (-2/m) * X.T @ error # (m,n) (n,m) @ (m,) = (n,)

            w = w - alpha * grad 
        elif method == 'stochastic':
            # we loop over each element and then update
            for i in range(m):
                xi = X[i, :].reshape(1, -1) # (1,n)
                yi = y[i] # (1)

                y_hat = xi @ w # (1,n) @ n = (1)

                error = y_hat - y # (1)

                grad = -2 * xi.T @ error # (1,n) -> (n,1) @ 1 = (n)

                w = w - alpha * grad 
        elif method == 'mini_batch':
            for i in range(0, m , batch_size):
                X_batch = X[i:i+batch_size] # (b,n)
                y_batch = y[i:i+batch_size] # (b)

                y_hat = X_batch @ w # (b,n) @ (n,) = (b,)

                error = y_hat - y_batch # (b,)

                grad = (-2/batch_size) * (X_batch.T @ error) # (n,b) @ (b)

                w = w - alpha * grad 

    return w 