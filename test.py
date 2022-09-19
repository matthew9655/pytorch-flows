import numpy as np
import numpy.random as npr
import datasets
import matplotlib.pyplot as plt

# def gen_bimodal(num_samples, N, sigma, rng):
#         theta1 = rng.uniform(0, 3, num_samples)
#         theta2 = rng.uniform(0, 3, num_samples)
#         y = np.stack((theta1,theta2), axis=1)

#         x = np.zeros((num_samples, N * 2))
#         for i in range(num_samples):
#             X1 = rng.normal(theta1[i], sigma, N)
#             X2 = rng.normal(theta2[i], sigma, N)
#             x[i] = np.concatenate([X1, X2])

#         return x, y
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    # x, y = gen_bimodal(100, 300, 0.25, rng)
    # plt.figure()
    # plt.axvline(y[0, 0], c='r')
    # plt.axvline(y[0, 1], c='r')
    # plt.hist(x[0, :], bins=50)
    # plt.savefig('test')
    dataset = getattr(datasets, 'Bimodal')(sigma=True)
    print(dataset.trn.x.shape)
    print(dataset.trn.y.shape)
    print(dataset.trn.y.shape[1])

    rand = rng.randint(0, 25000, 1)

    for i in rand:
        plt.figure()
        plt.axvline(dataset.trn.y[i, 0], c='r')
        plt.axvline(dataset.trn.y[i, 1], c='r')
        plt.title(f"mu1: {dataset.trn.y[i, 0]}")
        plt.hist(dataset.trn.x[i], bins=50)
        plt.savefig(f"dataset_plots/hist_{i}")
        plt.close()
