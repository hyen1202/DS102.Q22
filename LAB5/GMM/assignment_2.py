import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from scr.gmm import GMM

gmm = GMM(n_components = 2, max_iter=100, comp_names = ['0', '1'])

img_path = os.path.join(current_dir, "cow.jpg")
image = Image.open(img_path).convert("RGB")
image = np.array(image)

H, W, C = image.shape
X = image.reshape(H*W, C)

gmm.fit(X)

predicted = gmm.predict(X)
numeric_predicted = np.array(predicted, dtype=int)
mask = numeric_predicted.reshape(H, W)

plt.imshow(mask, cmap='gray')
plt.show()