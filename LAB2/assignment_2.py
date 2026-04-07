import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_mnist
from src.softmax_regression import SoftmaxRegression

def convert_to_onehot_vector(labels: np.ndarray):
    N = labels.shape[0]
    total_classes = labels.max() + 1
    oh_labels = np.zeros((N, total_classes))
    oh_labels[np.arange(N), labels] = 1
    return oh_labels

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

#labels one-hot encoding
train_labels_oh = convert_to_onehot_vector(train_labels)

#train
model = SoftmaxRegression(epoch = 1000, lr = 0.1)
model.fit(train_images, train_labels_oh)

#predict
predict_probs = model.predict(test_images) #xác suất
predict_labels = np.argmax(predict_probs, axis=1) #nhãn dự đoán

#evaluate
metrics = model.evaluate(test_labels, predict_labels)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-score: {metrics['f1']:.4f}")

#visulize loss function
plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()