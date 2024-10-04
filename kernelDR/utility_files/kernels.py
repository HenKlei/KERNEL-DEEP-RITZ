#!/usr/bin/env python3

from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt

# Abstract kernel
class Kernel(ABC):
    @abstractmethod    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def eval(self, x, y):
        pass

    def eval_prod(self, x, y, v, batch_size=100):
        N = x.shape[0]
        num_batches = int(np.ceil(N / batch_size))
        mat_vec_prod = np.zeros((N, 1)) 
        for idx in range(num_batches):
            idx_begin = idx * batch_size
            idx_end = (idx + 1) * batch_size
            A = self.eval(x[idx_begin:idx_end, :], y)
            mat_vec_prod[idx_begin:idx_end] = A @ v
        return mat_vec_prod

    @abstractmethod
    def diagonal(self, X):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

# Abstract RBF
class RBF(Kernel):
    @abstractmethod    
    def __init__(self):
        super(RBF, self).__init__()
        
    def eval(self, x, y):
        return self.rbf(self.ep, distance_matrix(np.atleast_2d(x), np.atleast_2d(y)))

    def diagonal(self, X):
        return np.ones(X.shape[0]) * self.rbf(self.ep, 0)
    
    def __str__(self):
     return self.name + ' [gamma = %2.2e]' % self.ep   

    def set_params(self, par):
        self.ep = par

# Implementation of concrete RBFs
class Gaussian(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'gauss'
        self.rbf = lambda ep, r: np.exp(-(ep * r) ** 2)

class GaussianTanh(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'gauss_tanh'
        self.rbf = lambda ep, r: np.exp(-(ep * np.tanh(r)) ** 2)

class IMQ(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'imq'
        self.rbf = lambda ep, r: 1. / np.sqrt(1 + (ep * r) ** 2)

class Matern(RBF):
    def __init__(self, ep=1, k=0):
        self.ep = ep
        if k == 0:
            self.name = 'mat0'
            self.rbf = lambda ep, r : np.exp(-ep * r)
        elif k == -1:
            self.name = 'derivative kernel ob quadratic matern'
            self.rbf = lambda ep, r: np.exp(-r) * (r**2 - (2 * 1 + 3) * r + 1 ** 2 + 2 * 1)
        elif k == 1:
            self.name = 'mat1'
            self.rbf = lambda ep, r: np.exp(-ep * r) * (1 + ep * r)
        elif k == 2:
            self.name = 'mat2'
            self.rbf = lambda ep, r: np.exp(-ep * r) * (3 + 3 * ep * r + 1 * (ep * r) ** 2)
        elif k == 3:
            self.name = 'mat3'
            self.rbf = lambda ep, r: np.exp(-ep * r) * (15 + 15 * ep * r + 6 * (ep * r) ** 2 + 1 * (ep * r) ** 3)
        elif k == 4:
            self.name = 'mat4'
            self.rbf = lambda ep, r: np.exp(-ep * r) * (105 + 105 * ep * r + 45 * (ep * r) ** 2 + 10 * (ep * r) ** 3 + 1 * (ep * r) ** 4)
        elif k == 5:
            self.name = 'mat5'
            self.rbf = lambda ep, r: np.exp(-ep * r) * (945 + 945 * ep * r + 420 * (ep * r) ** 2 + 105 * (ep * r) ** 3 + 15 * (ep * r) ** 4 + 1 * (ep * r) ** 5)
        elif k == 6:
            self.name = 'mat6'
            self.rbf = lambda ep, r: np.exp(-ep * r) * (10395 + 10395 * ep * r + 4725 * (ep * r) ** 2 + 1260 * (ep * r) ** 3 + 210 * (ep * r) ** 4 + 21 * (ep * r) ** 5 + 1 * (ep * r) ** 6)
        elif k == 7:
            self.name = 'mat7'
            self.rbf = lambda ep, r: np.exp(-ep * r) * (135135 + 135135 * ep * r + 62370 * (ep * r) ** 2 + 17325 * (ep * r) ** 3 + 3150 * (ep * r) ** 4 + 378 * (ep * r) ** 5 + 28 * (ep * r) ** 6 + 1 * (ep * r) ** 7)
        else:
            self.name = None
            self.rbf = None
            raise Exception('This Matern kernel is not implemented')

class Wendland(RBF):
    def __init__(self, ep=1, k=0, d=1):
        self.ep = ep
        self.name = 'wen_' + str(d) + '_' + str(k)
        l = int(np.floor(d / 2) + k + 1)
        if k == 0:
            p = lambda r: 1
        elif k == 1:
            p = lambda r: (l + 1) * r + 1
        elif k == 2:
            p = lambda r: (l + 3) * (l + 1) * r ** 2 + 3 * (l + 2) * r + 3
        elif k == 3:
            p = lambda r: (l + 5) * (l + 3) * (l + 1) * r ** 3 + (45 + 6 * l * (l + 6)) * r ** 2 + (15 * (l + 3)) * r + 15
        elif k == 4:
            p = lambda r: (l + 7) * (l + 5) * (l + 3) * (l + 1) * r ** 4 + (5 * (l + 4) * (21 + 2 * l * (8 + l))) * r ** 3 + (45 * (14 + l * (l + 8))) * r ** 2 + (105 * (l + 4)) * r + 105
        else:
            raise Exception('This Wendland kernel is not implemented')
        c = np.math.factorial(l + 2 * k) / np.math.factorial(l)
        e = l + k
        self.rbf = lambda ep, r: np.maximum(1 - ep * r, 0) ** e * p(ep * r) / c

class MQ(RBF):
    def __init__(self, ep=1, beta=1.5):
        self.ep = ep
        self.beta = beta
        assert beta != int(beta)

        self.name = 'mq'
        self.rbf = lambda ep, r: (1 + (ep * r) ** 2) ** self.beta
        self.m = max([0, np.ceil(self.beta)])

 # Polynomial kernels    
class Polynomial(Kernel):
    def __init__(self, a=0, p=1):
        self.a = a
        self.p = p
            
    def eval(self, x, y):
        return (np.atleast_2d(x) @ np.atleast_2d(y).transpose() + self.a) ** self.p
    
    def diagonal(self, X):
        return ((np.linalg.norm(X, axis=1)**2 + self.a) ** self.p) #[:, None]

    def __str__(self):
     return 'polynomial' + ' [a = %2.2e, p = %2.2e]' % (self.a, self.p)   

    def set_params(self, par):
        self.a = par[0]
        self.p = par[1]


# Polynomial kernels
class BrownianBridge(Kernel):
    def __init__(self):
        super().__init__()
        self.name = 'Brownian Bridge'

    def eval(self, x, y):
        return np.minimum(np.atleast_2d(x), np.transpose(np.atleast_2d(y))) - np.atleast_2d(x) * np.transpose(np.atleast_2d(y))
    
    def d1_eval(self, x, y):            # derivative wrt first argument

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        array1 = (x <= y.T) * (1 - y.T)
        array2 = (x > y.T)  * (-y.T)


        return array1 + array2       # no clue why we need this minus here, does not appear in papers


    def diagonal(self, X):
        return X[:, 0] - X[:, 0] ** 2

    def __str__(self):
        return 'Brownian Bridge kernel'

    def set_params(self, par):
        pass


class weighted_basic_Matern(Kernel):
    # 1D kernel!!!

    def __init__(self):
        super().__init__()
        self.name = 'basic_Matern'

        self.base_kernel = Matern(ep=1, k=0)

        self.b_func = lambda x: x*(1-x)
        self.b_func1 = lambda x: 1-2*x

    def eval(self, x, y):
        return self.base_kernel.eval(x, y) * self.b_func(x) * self.b_func(y.T)

    
    def d1_eval(self, x, y):            # derivative wrt first argument

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)


        # see 05.07.24\A2
        array1 = self.b_func1(x) * self.base_kernel.eval(x, y) * self.b_func(y.T)
        array2 = self.b_func(x) * np.sign(y.T-x) * np.exp(-np.abs(x-y.T)) * self.b_func(y.T)

        return array1 + array2
    
    def diagonal(self, X):
        return self.b_func(X)**2 * self.base_kernel.diagonal(X)

    def __str__(self):
        return 'Brownian Bridge kernel'

    def set_params(self, par):
        pass


class ConvBrownianBridge(Kernel):
    def __init__(self):
        super().__init__()
        self.name = 'Convolution Brownian Bridge'

    def eval(self, x, y):

        x = np.atleast_2d(x)
        z = np.atleast_2d(y)

        array1 = (x <= z.T) * (1/6 * x * (1-z.T) * (x**2 + z.T**2 - 2*z.T))
        array2 = (z.T <  x)  * (1/6 * (1-x) * z.T * (x**2 + z.T**2 - 2*x))


        return -(array1 + array2)       # no clue why we need this minus here, does not appear in papers

    def diagonal(self, X):
        return X[:, 0] - X[:, 0] ** 2

    def __str__(self):
        return 'Brownian Bridge kernel'

    def set_params(self, par):
        pass

class BrownianMotion(Kernel):
    def __init__(self):
        super().__init__()
        self.name = 'Brownian Motion'

    def eval(self, x, y):
        return np.minimum(np.atleast_2d(x), np.transpose(np.atleast_2d(y)))

    def diagonal(self, X):


        return X.reshape(-1)

    def __str__(self):
        return 'Brownian Motion kernel'

    def set_params(self, par):
        pass

# Convolution kernel for Wendland k=0 kernel max(0, 1-|x-y|) on [0,1]. For calculation see 11.02.23\A1.
class ConvKernelWendland(Kernel):
    def __init__(self):
        super().__init__()
        self.name = 'Convolution Kernel Wendland k=0'

        self.func_util = lambda x, y: 1/3 * (-x**3 + 3*x**2*y - 3*x**2 - 3*x*y**2 + 3*x*y + 3/2*x + y**3 - 3*y**2 + 3/2*y + 1)

    def eval(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        A1 = self.func_util(x, y.transpose())
        A2 = self.func_util(y, x.transpose()).transpose()

        # Calculate matrix of logicals to check where x < y
        diff = x - y.transpose()
        M = diff <= 0

        return A1 * M + A2 * ~M

    def diagonal(self, X):
        return 1/3 + X[:, 0] - X[:, 0] ** 2

    def __str__(self):
        return 'Convolution Kernel Wendland k=0'

    def set_params(self, par):
        pass

# Convolution kernel for linear Matern kernel on [0,1]. For calculation see 02.01.24\C1.
class ConvKernelLinMatern(Kernel):
    def __init__(self):
        super().__init__()
        self.name = 'Convolution Kernel linear Matern'


        self.func_util1 = lambda x, z: 1/2 * np.exp(-z) * (((x+3)*z + 5) * np.sinh(x) - x * (z+3) * np.cosh(x))
        self.func_util2 = lambda x, z: -1/6 * np.exp(x-z) * (x-z) * (x**2 - 2*x*(z+3) + z*(z+6) + 6)
        self.func_util3 = lambda x, z: 1/4 * np.exp(x-z) * (np.exp(2*z-2) * (x*(5-2*z) + 5*z - 13) - 3*x + 3*z + 5)

        self.func_util = lambda x, y: self.func_util1(x, y) + self.func_util2(x, y) + self.func_util3(x, y)

    def eval(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        A1 = self.func_util(x, y.transpose())
        A2 = self.func_util(y, x.transpose()).transpose()

        # Calculate matrix of logicals to check where x < y
        diff = x - y.transpose()
        M = diff <= 0

        return A1 * M + A2 * ~M

    def diagonal(self, X):
        part1 = 1/2 * np.exp(-X[:, 0]) * (((X[:, 0] + 3) * X[:, 0] + 5) * np.sinh(X[:, 0]) + X[:, 0]*(X[:, 0]+3) * np.cosh(X[:, 0]))
        part2 = 1/4 * (np.exp(2*X[:, 0]-2) * (X[:, 0] * (5-2*X[:, 0]) + 5*X[:, 0] - 13) + 5)

        return part1 + part2

    def __str__(self):
        return 'Convolution Kernel linear Matern'

    def set_params(self, par):
        pass


# Tensor product kernels
class TensorProductKernel(Kernel):
    def __init__(self, kernel):
        super().__init__()

        self.kernel_1D = kernel
        self.name = self.kernel_1D.name

    def eval(self, x, y):

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        assert x.shape[1] == y.shape[1], 'Dimension do not match'

        array_matrix = np.ones((x.shape[0], y.shape[0]))

        for idx_dim in range(x.shape[1]):
            array_matrix = array_matrix * self.kernel_1D.eval(x[:, [idx_dim]], y[:, [idx_dim]])

        return array_matrix

    def diagonal(self, X):

        X = np.atleast_2d(X)

        array_diagonal = np.ones(X.shape[0])

        for idx_dim in range(X.shape[1]):
            array_diagonal *= self.kernel_1D.diagonal(X[:, [idx_dim]])

        return array_diagonal

    def __str__(self):
        return 'Tensor product kernel for ' + self.name

    def set_params(self, par):
        pass

class TensorProductKernel_list(Kernel):
    def __init__(self, list_kernel):
        super().__init__()

        self.list_kernels = list_kernel
        self.name = 'Tensor Product Kernel'
 

    def eval(self, x, y):

        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        assert x.shape[1] == y.shape[1], 'Dimension do not match'

        array_matrix = np.ones((x.shape[0], y.shape[0]))

        for idx_dim in range(x.shape[1]):
            array_matrix = array_matrix * self.list_kernels[idx_dim].eval(x[:, [idx_dim]], y[:, [idx_dim]])

        return array_matrix

    def diagonal(self, X):

        X = np.atleast_2d(X)

        array_diagonal = np.ones(X.shape[0])

        for idx_dim in range(X.shape[1]):
            array_diagonal *= self.list_kernels[idx_dim].diagonal(X[:, [idx_dim]])

        return array_diagonal

    def __str__(self):
        return 'Tensor product kernel for ' + self.name

    def set_params(self, par):
        pass

# A demo usage
def main():
    ker = Gaussian()

    x = np.linspace(-1, 1, 100)[:, None]
    y = np.matrix([0])
    A = ker.eval(x, y)


    fig = plt.figure(1)
    fig.clf()
    ax = fig.gca()
    ax.plot(x, A)
    ax.set_title('A kernel: ' + str(ker))
    fig.show()


if __name__ == '__main__':
    main()


        
