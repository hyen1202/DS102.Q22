import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_mnist
from src.logistic_regression import LogisticRegression

#load data
train_images, train_labels, test_images, test_labels = load_mnist()

#normalize
train_images = train_images.astype(np.float32) / 255.0
test_images  = test_images.astype(np.float32) / 255.0

#flatten images
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1)
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1) 

# Filter only digits 0 and 1
mask_train = np.isin(train_labels, [0, 1])
mask_test  = np.isin(test_labels,  [0, 1])

X_train = train_images[mask_train]
y_train = train_labels[mask_train]
X_test  = test_images[mask_test]
y_test  = test_labels[mask_test]

#train
model = LogisticRegression(epoch = 1000, lr = 0.01)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#evaluate
y_pred = (y_pred >= 0.5).astype(int).flatten()
y_test = y_test.reshape(-1)
metrics = model.evaluate(y_test, y_pred)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-score: {metrics['f1']:.4f}")

#visulize loss function
plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()