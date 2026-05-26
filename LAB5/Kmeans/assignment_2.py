import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from scr.data_loader import generate_data, display
from scr.kmeans import Kmeans

sigma = np.eye(2)

X, y = generate_data([
    (1200, [2, 2], sigma),
    (200, [8, 3], sigma),
    (1000, [3, 6], sigma)
])

display(X, y, title='Dữ liệu ban đầu')

model = Kmeans(3)
centroids, points = model.fit(X, iterations = 15)
print('Centroids found by kmeans:\n', centroids)

display(X, points, 'Dữ liệu sau khi phân cụm')
plt.show()