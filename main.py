"""
============================================================
Tensor Operations in Python Examples
============================================================
The following demonstrates naive tensor operations in Python
"""
import numpy as np


class TensorOpsInPython:
    def __init__(self):
        print(__doc__)
        self.array_1d_y1 = [0, 1, 2, 3, 4, 5]
        self.array_1d_x1 = [9, 7, 5, 3, 1, 0]

        self.array_2d_x = [[0, 1, 2], [3, 4, 5]]
        self.array_2d_y = [[2, 4, 6], [8, 10, 12]]

        self.array_2d_x1 = [[0, 1, 2, 3, 4, 5],
                            [1, 2, 3, 4, 5, 6],
                            [2, 3, 4, 5, 6, 7]]

        self.array_2d_y1 = [[3, 2, 1],
                            [4, 2, 0],
                            [0, 3, 1],
                            [0, 1, 0],
                            [2, 4, 1],
                            [4, 4, 1]]

        self.tensor_1d_x1 = np.array(self.array_1d_x1)
        self.tensor_1d_y1 = np.array(self.array_1d_y1)

        self.tensor_2d_x = np.array(self.array_2d_x)
        self.tensor_2d_y = np.array(self.array_2d_y)

        self.tensor_2d_x1 = np.array(self.array_2d_x1)
        self.tensor_2d_y1 = np.array(self.array_2d_y1)

        # Naive Relu
        print('\n--- Naive Relu')
        print('Input')
        print(self.tensor_2d_x)
        x = self.naive_relu(self.tensor_2d_x)
        print('\nOutput')
        print(x)
        print('\nNumpy')
        x = np.maximum(self.tensor_2d_x,0)
        print(x)

        # Naive Add
        print('\n--- Naive Add')
        print('Input')
        print(self.tensor_2d_x)
        print(self.tensor_2d_y)
        x = self.naive_add(self.tensor_2d_x, self.tensor_2d_y)
        print('\nOutput')
        print(x)
        print('\nNumpy')
        x = self.tensor_2d_x + self.tensor_2d_y
        print(x)

        # Naive Subtract
        print('\n--- Naive Subtract')
        print('Input')
        print(self.tensor_2d_x)
        print(self.tensor_2d_y)
        x = self.naive_subtract(self.tensor_2d_x, self.tensor_2d_y)
        print('\nOutput')
        print(x)
        print('\nNumpy')
        x = self.tensor_2d_x - self.tensor_2d_y
        print(x)

        # Naive Multiply
        print('\n--- Naive Multiply')
        print('Input')
        print(self.tensor_2d_x)
        print(self.tensor_2d_y)
        x = self.naive_multiply(self.tensor_2d_x, self.tensor_2d_y)
        print('\nOutput')
        print(x)
        print('\nNumpy')
        x = self.tensor_2d_x * self.tensor_2d_y
        print(x)

        # Naive Add Matrix and Vector
        print('\n--- Naive Add Matrix and Vector')
        print('Input')
        print(self.tensor_2d_x1)
        print(self.tensor_1d_y1)
        x = self.naive_add_matrix_and_vector(self.tensor_2d_x1, self.tensor_1d_y1)
        print('\nOutput')
        print(x)

        # Naive Vector Dot (product)
        print('\n--- Naive Vector Dot (product)')
        print('Input')
        print(self.tensor_1d_x1)
        print(self.tensor_1d_y1)
        x = self.naive_vector_dot(self.tensor_1d_x1, self.tensor_1d_y1)
        print('\nOutput')
        print(x)

        # Naive Matrix Vector Dot (product)
        print('\n--- Naive Vector Dot (product)')
        print('Input')
        print(self.tensor_2d_x1)
        print(self.tensor_1d_y1)
        x = self.naive_matrix_vector_dot(self.tensor_2d_x1, self.tensor_1d_y1)
        print('\nOutput')
        print(x)

        # Naive Matrix Vector Dot v2 reusing vector dot
        print('\n--- Naive Vector Dot V2 reusing vector dot(product)')
        print('Input')
        print(self.tensor_2d_x1)
        print(self.tensor_1d_y1)
        x = self.naive_matrix_vector_dot_v2(self.tensor_2d_x1, self.tensor_1d_y1)
        print('\nOutput')
        print(x)

        # Naive Matrix Dot
        print('\n--- Naive Matrix Dot')
        print('Input')
        print(self.tensor_2d_x1)
        print(self.tensor_2d_y1)
        x = self.naive_matrix_dot(self.tensor_2d_x1, self.tensor_2d_y1)
        print('\nOutput')
        print(x)

        # Reshape
        print('\n--- Reshape')
        print('Input')
        print(self.tensor_2d_x)
        x = self.tensor_2d_x.reshape((3, 2))
        print('\nOutput after reshape to 3, 2')
        print(x)

        # Transpose
        print('\n--- Transpose (exchanging rows and columns)')
        print('Input')
        print(self.tensor_2d_x)
        x = np.transpose(self.tensor_2d_x)
        print('\nOutput after transpose')
        print(x)

    @staticmethod
    def naive_relu(x):
        assert len(x.shape) == 2
        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = max(x[i, j], 0)
        return x

    @staticmethod
    def naive_add(x, y):
        assert len(x.shape) == 2
        assert x.shape == y.shape
        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += y[i, j]
        return x

    @staticmethod
    def naive_subtract(x, y):
        assert len(x.shape) == 2
        assert x.shape == y.shape
        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] -= y[i, j]
        return x

    @staticmethod
    def naive_multiply(x, y):
        assert len(x.shape) == 2
        assert x.shape == y.shape
        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] *= y[i, j]
        return x

    # Example broadcasting axes to smaller tensor to match ndim of larger tensor,
    # so smaller tensor is repeated along new axes to match full shape of larger tensor
    @staticmethod
    def naive_add_matrix_and_vector(x, y):
        assert len(x.shape) == 2
        assert len(y.shape) == 1
        assert x.shape[1] == y.shape[0]

        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += y[j]
        return x

    # Only works with vectors having same number elements
    @staticmethod
    def naive_vector_dot(x, y):
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert x.shape[0] == y.shape[0]
        z = 0
        for i in range(x.shape[0]):
            z += x[i] * y[i]
        return z

    # Only works with vectors having same number elements
    @staticmethod
    def naive_matrix_vector_dot(x, y):
        assert len(x.shape) == 2  # Numpy matrix
        assert len(y.shape) == 1  # Numpy vector
        assert x.shape[1] == y.shape[0]

        z = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i] += x[i, j] * y[j]
        return z

    def naive_matrix_vector_dot_v2(self, x, y):
        z = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            z[i] = self.naive_vector_dot(x[i, :], y)
        return z

    # Dot product between 2 matrices. Result is matrix with shale x.shape[0], y.shape[1],
    # where coefficients are the vector products between rows of x and columns of y
    def naive_matrix_dot(self, x, y):
        assert len(x.shape) == 2  # Numpy matrix
        assert len(y.shape) == 2  # Numpy matrix
        assert x.shape[1] == y.shape[0]

        z = np.zeros((x.shape[0], y.shape[1]))
        for i in range(x.shape[0]):
            for j in range(y.shape[1]):
                row_x = x[i, :]
                column_y = y[:, j]
                z[i, j] = self.naive_vector_dot(row_x, column_y)
        return z


tp = TensorOpsInPython()
