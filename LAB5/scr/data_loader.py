import numpy as np
import matplotlib.pyplot as plt


def generate_data(cluster_configs):
    # cluster_configs: list of (n_samples, mean, cov)
    X = np.vstack([np.random.multivariate_normal(mean, cov, n)
         for n, mean, cov in cluster_configs])
    y = np.concatenate([
        np.full(n, k) for k, (n, _, _) in enumerate(cluster_configs)
    ])
    return X, y

def display(X, label, title='Data visualization'):
    plt.figure()
    
    label = np.asarray(label)
    unique_labels = np.unique(label)

    kwargs = {'markersize':5, 'alpha':.8, 'markeredgecolor':'k'}
    colors = ['r', 'b', 'g', 'c', 'm', 'y']
    markers = ['o', '^', 's', 'D', 'X', 'P']
    for idx, i in enumerate(unique_labels):
        X_i = X[label == i, :]
        color_marker = colors[idx % len(colors)] + markers[idx % len(markers)]
        plt.plot(X_i[:, 0], X_i[:, 1], color_marker, **kwargs)
    plt.title(title)
    