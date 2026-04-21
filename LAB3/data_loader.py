import os
import cv2
import numpy as np

def load_data(data_dir):
    X = []
    y = []

    l = [(-1, 'NORMAL'), (1, 'PNEUMONIA')]
    for label, class_name in l:
        class_dir = os.path.join(data_dir, class_name)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))

            img = img.flatten().astype(np.float32)

            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y, dtype = np.int32)

    return X, y




