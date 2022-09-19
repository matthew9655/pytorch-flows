import os

import matplotlib.pyplot as plt
import numpy as np
from flows import BatchNormFlow
import torch
import torchvision
from sklearn.decomposition import PCA


def save_moons_plot(epoch, best_model, dataset):
    # generate some examples
    best_model.eval()
    with torch.no_grad():
        x_synth = best_model.sample(500).detach().cpu().numpy()

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(dataset.val.x[:, 0], dataset.val.x[:, 1], '.')
    ax.set_title('Real data')

    ax = fig.add_subplot(122)
    ax.plot(x_synth[:, 0], x_synth[:, 1], '.')
    ax.set_title('Synth data')

    try:
        os.makedirs('plots')
    except OSError:
        pass

    plt.savefig('plots/plot_{:03d}.png'.format(epoch))
    plt.close()


batch_size = 100
fixed_noise = torch.Tensor(batch_size, 28 * 28).normal_()
y = torch.arange(batch_size).unsqueeze(-1) % 10
y_onehot = torch.FloatTensor(batch_size, 10)
y_onehot.zero_()
y_onehot.scatter_(1, y, 1)


def save_images(epoch, best_model, cond):
    best_model.eval()
    with torch.no_grad():
        if cond:
            imgs = best_model.sample(batch_size, noise=fixed_noise, cond_inputs=y_onehot).detach().cpu()
        else:
            imgs = best_model.sample(batch_size, noise=fixed_noise).detach().cpu()

        imgs = torch.sigmoid(imgs.view(batch_size, 1, 28, 28))
    
    try:
        os.makedirs('images')
    except OSError:
        pass

    torchvision.utils.save_image(imgs, 'images/img_{:03d}.png'.format(epoch), nrow=10)

def plot_pca(mat, n_components, epoch):
    pca = PCA(n_components)
    reduced = pca.fit_transform(mat)
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], s=3)
    plt.savefig(f'scatter_plots/{epoch}')
    plt.close()

def sample_dist(epoch, best_model, sigma=False):
    best_model.eval()
    rng = np.random.RandomState(42)
    batch_size = 10
    noise = torch.zeros(batch_size, 500)
    theta1 = rng.uniform(0, 3, batch_size)
    theta2 = rng.uniform(0, 3, batch_size)
    if sigma:
        sig1 = rng.uniform(0.15, 1, batch_size)
        sig2 = rng.uniform(0.15, 1, batch_size)
        y = np.stack((theta1,theta2, sig1, sig2), axis=1)
    else:
        y = np.stack((theta1,theta2), axis=1)

    y = torch.from_numpy(y).float()
    with torch.no_grad():
        dists = best_model.sample(batch_size, noise=noise, cond_inputs=y).detach().cpu()
        
    try:
        os.makedirs('dist_images')
    except OSError:
        pass

    rng = np.random.RandomState(42)
    
    for i in range(batch_size):
        plt.figure()
        plt.axvline(y[i, 0], c='r')
        plt.axvline(y[i, 1], c='r')
        plt.hist(dists[i, :], bins=50)
        plt.savefig(f'dist_images/distsample_{epoch}_{i}')
        plt.close()


if __name__ == "__main__":
    pass
