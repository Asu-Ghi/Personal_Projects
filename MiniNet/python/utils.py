from nn_bindings import *

# Returns a matrix object from a numpy array
def numpy_to_matrix(array):
    rows, cols = array.shape
    array_ptr = array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # Matrix->data
    return matrix(array_ptr, ctypes.byref(ctypes.c_int(rows), ctypes.byref(ctypes.c_int(cols)))) # (Matrix->dim1, Matrix->dim2)
