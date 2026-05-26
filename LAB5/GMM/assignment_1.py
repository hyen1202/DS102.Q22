import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from scr.data_loader import generate_data, display
from scr.gmm import GMM

sigma = np.eye(2)
sigma2 = np.array([[10, 0], [0, 1]])

X1, y1 = generate_data([
    (200, [2, 2], sigma),
    (200, [8, 3], sigma),
    (200, [3, 6], sigma)
])

X2, y2 = generate_data([
    (1200, [2, 2], sigma),
    (200, [8, 3], sigma),
    (1000, [3, 6], sigma)
])

X3, y3 = generate_data([
    (200, [2, 2], sigma),
    (200, [8, 3], sigma),
    (200, [3, 6], sigma2)
])

display(X1, y1, title="GMM - Dataset 1 ban đầu")
model1 = GMM(n_components=3)
model1.fit(X1)
cluster1 = model1.predict(X1)
display(X1, cluster1, title="GMM - Dataset 1 sau khi phân cụm")

display(X2, y2, title="GMM - Dataset 2 ban đầu")
model2 = GMM(n_components=3)
model2.fit(X2)
cluster2 = model2.predict(X2)
display(X2, cluster2, title="GMM - Dataset 2 sau khi gom cụm")

display(X3, y3, title="GMM - Dataset 3 ban đầu")
model3 = GMM(n_components=3)
model3.fit(X3)
cluster3 = model3.predict(X3)
display(X3, cluster3, title="GMM - Dataset 3 sau khi gom cụm")
plt.show()