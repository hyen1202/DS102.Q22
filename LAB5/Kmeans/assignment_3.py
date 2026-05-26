import numpy as np
import matplotlib.pyplot as plt
import os
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from scr.data_loader import generate_data, display
from scr.kmeans import Kmeans

sigma1 = np.eye(2)
sigma2 = np.array(
    [[10, 0],
    [0, 1]]
)

X, y = generate_data([
    (200, [2, 2], sigma1),
    (200, [8, 3], sigma1),
    (200, [3, 6], sigma2)
])

display(X, y, 'Dữ liệu ban đầu')

model = Kmeans(3)
centroids, points = model.fit(X, iterations = 15)
print('Centroids found by kmeans:\n', centroids)

display(X, points, 'Dữ liệu sau khi phân cụm')
plt.show()