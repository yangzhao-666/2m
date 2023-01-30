import numpy as np

def dot(a,b):
    length = b.shape[1]
    c = np.zeros(length)
    for i in np.arange(length):
        c[i] = np.inner(a,b[:,i])
    return c

while True:
    state = np.random.uniform(size=(10, 10, 6))
    rp_matrix = np.random.uniform(size=(600, 16))
    #print(dot(state.flatten(), rp_matrix))
    print(dot(state.flatten(), rp_matrix))
    #print(np.matmul(state.flatten(), rp_matrix))
    #print(state.flatten())
