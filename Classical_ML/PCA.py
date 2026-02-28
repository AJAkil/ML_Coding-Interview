import numpy as np 

def PCA(data, k):
    # first standardize the data
    # data (m, n)

    # standardize the data first
    data_ = (data - np.mean(data, axis=0) / np.std(data, axis=0))

    data_cov = np.cov(data_, rowvar=False) # (n, n)

    eigenvalues, eigenvectors = np.linalg.eigh(data_cov)
    # eigenvalues (n)
    # eigenvectors (n, n) column vectors

    # then we reorder the eigenvalues first since we need the highest eigenvalues
    indices = np.argsort(eigenvalues)[::-1] # (n)

    # then we sort the eigenvectors themselves here
    eigenvectors = eigenvectors[:, indices]  # n,n but sorted acc to eigenvalues

    # then we take the topk pc
    components = eigenvectors[:, k] # since column wise vectors

    # we can also do the sign convention one here
    for j in range(components.shape[1]):
        mask = np.abs(components[:, j]) > 1e-10

        if np.any(mask): # checks if there is a 1 somewhere
            # gets the index now
            index = np.where(mask)[0][0]

            # then we check if that element is < 0
            if components[index, j] < 0:
                # then we multiply the entire component by -1
                components[:, j] *= -1


    return np.round(components, 4)

