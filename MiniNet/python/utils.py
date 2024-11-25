from nn_bindings import *

# Returns a matrix object from a numpy array
def numpy_to_matrix(df):
    if isinstance(df, pd.DataFrame):
        df = df.values # Converts Pandas Df to Numpy

    data = df.flatten()  # Flatten DataFrame to 1D numpy array
    matrix_obj = matrix()
    matrix_obj.data = (ctypes.c_double * len(data))(*data)  # Convert array to ctypes array
    matrix_obj.dim1 = df.shape[0]
    matrix_obj.dim2 = df.shape[1]
    return matrix_obj

# Returns numpy array from matrix
def matrix_to_numpy(mat):
    shape = (mat.dim1, mat.dim2)
    # Convert ctypes array to a numpy array
    data_array = np.ctypeslib.as_array(mat.data, shape)
    
    # Reshape the numpy array to match the matrix dimensions
    return data_array