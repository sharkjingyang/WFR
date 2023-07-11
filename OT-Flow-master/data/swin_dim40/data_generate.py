import numpy as np
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import matplotlib.pyplot as plt

data_name="8gaussians"


def sample_generate(data_name,rng=None,batch_size=20000):
    if rng is None:
        rng = np.random.RandomState()

    if data_name == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data



    elif data_name == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data_name == "rings":
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
        return X

    elif data_name == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data
    
    elif data_name == "8gaussians":
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
    
        return dataset
    
data_minibone = np.load('C:/Users/shark/桌面/OT-Flow-master/data/miniboone/data.npy')
# mu = data_minibone.mean(axis=0)
# s = data_minibone.std(axis=0)
# data_minibone = (data_minibone - mu) / s
data_2d=sample_generate(data_name,rng=None,batch_size=data_minibone.shape[0])
data=data_minibone.copy()
data[:,0]=data_2d[:,0]
data[:,1]=data_2d[:,1]
mu = data.mean(axis=0)
s = data.std(axis=0)
data = (data - mu) / s

nBins=70
plt.hist2d(data[:,0],data[:,1], range=[[-3, 3], [-3, 3]], bins=nBins)
plt.show()

np.save("data/swin_dim40/data.npy",data)
