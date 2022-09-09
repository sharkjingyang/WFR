# From FFJORD
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import matplotlib.pyplot as plt
from math import *
from approximate_rho0 import *
# Dataset iterator

def density(x, centers, weights=[0.5,0.5]):
    return sum(1 / sqrt(2 * pi) * exp(-(x - i[0]) ** 2 / 2) * i[1] for i in zip(centers, weights))

def inf_train_gen(data, rng=None, batch_size=200, require_density=True, device=torch.device('cuda:' + str(1) if torch.cuda.is_available() else 'cpu')):
    if rng is None:
        rng = np.random.RandomState()
    
    if data == "fourweighted":
        num=int(batch_size/4)
        X_1=np.random.randn(num,2)
        X_1[:,0]=np.abs(X_1[:,0])
        X_1[:,1]=np.abs(X_1[:,1]) #1 xiangxian w=8

        X_2=np.random.randn(num,2)
        X_2[:,0]=np.abs(X_2[:,0])
        X_2[:,1]=np.abs(X_2[:,1]) #2 xiangxian w=2

        X_3=np.random.randn(num,2)
        X_3[:,0]=np.abs(X_3[:,0])
        X_3[:,1]=np.abs(X_3[:,1]) #3 xiangxian w=4

        X_4=np.random.randn(num,2)
        X_4[:,0]=np.abs(X_4[:,0])
        X_4[:,1]=np.abs(X_4[:,1])  #4 xiangxian w=6
        
        output_data=np.concatenate((X_1,X_2,X_3,X_4),axis=0)
        batch_one=np.ones((num,1))
        normalized_weight=np.concatenate((1.6*batch_one,0.4*batch_one,0.8*batch_one,1.2*batch_one),axis=0)
        return output_data,normalized_weight



    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5

        if require_density:
            p0, d_net = get_density(data, d=data.shape[1], niter=1, device=device)
            return data, p0, d_net
        else:
            return data


    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3

        if require_density:
            p0, d_net = get_density(data, d=data.shape[1], niter=1, device=device)
            return data, p0, d_net
        else:
            return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)
        X = X.astype("float32")

        if require_density:
            p0, d_net = get_density(X, d=X.shape[1], niter=1, device=device)
            return X, p0, d_net
        else:
            return X

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])

        if require_density:
            p0, d_net = get_density(data, d=data.shape[1], niter=1, device=device)
            return data, p0, d_net  
        else:
            return data


    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        epsilon = 0.0
        for i in range(int(batch_size*(1-epsilon))):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        for i in range(batch_size-int(batch_size*(1-epsilon))):
            point = rng.randn(2)*1.414*2
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414

        if require_density:
            p0, d_net = get_density(dataset, d=dataset.shape[1],niter=1,device=device)
            return dataset, p0, d_net
        else:
            return dataset

    elif data == "2gaussians":
        scale = 3.
        centers = [(1, 0), (-1, 0)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 2
            idx = rng.randint(2)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414

        if require_density:
            p0, d_net = get_density(dataset, d=dataset.shape[1], niter=1, device=device)
            return dataset, p0, d_net
        else:
            return dataset

    elif data == "1d":
        # centers = [-4, -1, 2, 5]
        # weights = [0.25, 0.25,0.25,0.25]
        centers = [-2, 2]
        weights = [1/2,1/2]

        dataset = []
        rhox_true = []
        for i in range(batch_size):
            point = rng.randn()
            idx = rng.randint(2)
            center = centers[idx]
            point += center
            dataset.append(point)
            rhox_true.append(density(point, centers, weights))
        dataset = np.array(dataset, dtype="float32")
        rhox_true = np.array(rhox_true, dtype="float32")
        if require_density:
            rhox = get_density(dataset, d=1, niter=1000,device=device)
            return dataset, rhox
        else:
            return dataset, rhox_true

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        dataset = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

        if require_density:
            p0, d_net = get_density(dataset, d=dataset.shape[1], niter=1, device=device)
            return dataset, p0, d_net
        else:
            return dataset


    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1


        if require_density:
            p0, d_net = get_density(x, d=x.shape[1],niter=1,device=device)
            return x, p0, d_net
        else:
            return x


    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        if require_density:
            p0, d_net = get_density(x, d=x.shape[1],niter=1,device=device)
            return x, p0, d_net
        else:
            return x



    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        dataset = np.stack((x, y), 1)
        if require_density:
            p0, d_net = get_density(dataset, d=dataset.shape[1], niter=1, device=device)
            return dataset, p0, d_net
        else:
            return dataset

    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        dataset = np.stack((x, y), 1)
        if require_density:
            p0, d_net = get_density(dataset, d=data.shape[1],niter=1,device=device)
            return dataset, p0, d_net
        else:
            return dataset
    else:
        return inf_train_gen("8gaussians", rng, batch_size)

if __name__ == "__main__":
    dataset, p1 = inf_train_gen('checkerboard', rng=None, batch_size=20000,require_density=True)