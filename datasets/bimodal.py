import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

import datasets

from . import util


class Bimodal:
    """
    The MNIST dataset of handwritten digits.
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):
            self.x = data[0]
            self.labels = data[1]  # numeric labels
            self.y = data[1]
            self.N = self.x.shape[0]  # number of datapoints
            self.x = self.x.astype('float32')
            self.y = self.y.astype('float32')

        # @staticmethod
        # def _dequantize(x, rng):
        #     """
        #     Adds noise to pixels to dequantize them.
        #     """
        #     return x + rng.rand(*x.shape) / 256.0

        # @staticmethod
        # def _logit_transform(x):
        #     """
        #     Transforms pixel values with logit to be unconstrained.
        #     """
        #     return util.logit(MNIST.alpha + (1 - 2 * MNIST.alpha) * x)

    def __init__(self, sigma=False):

        # load dataset
        N = 250

        rng = np.random.RandomState(42)
        trn = self._gen_bimodal(25000, N, rng, sigma)
        val = self._gen_bimodal(5000, N, rng, sigma)
        tst = self._gen_bimodal(3000, N, rng, sigma)

        
        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]

    @staticmethod
    def _gen_bimodal(num_samples, N, rng, sigma=False):
        theta1 = rng.uniform(0, 3, num_samples)
        theta2 = rng.uniform(0, 3, num_samples)

        if sigma:
            sig1 = rng.uniform(0.1, 0.5, num_samples)
            sig2 = rng.uniform(0.1, 0.5, num_samples) 
            y = np.stack((theta1,theta2, sig1, sig2), axis=1)
        else:
            y = np.stack((theta1,theta2), axis=1)

        x = np.zeros((num_samples, N * 2))
        for i in range(num_samples):
            if sigma:
                X1 = rng.normal(theta1[i], sig1[i], N)
                X2 = rng.normal(theta2[i], sig2[i], N)
                x[i] = np.concatenate([X1, X2])
            else:
                X1 = rng.normal(theta1[i], 0.15, N)
                X2 = rng.normal(theta2[i], 0.15, N)
                x[i] = np.concatenate([X1, X2])


        return x, y


if __name__ == "__main__":
    bi = Bimodal()
    print(bi.trn.x)
    print(bi.trn.y)
