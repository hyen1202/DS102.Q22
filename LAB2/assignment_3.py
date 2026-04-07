import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from src.data_loader import load_mnist

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

#-------------Logistic Regression----------
logistic_r = LogisticRegression(max_iter=1000, random_state=42)
logistic_r.fit(X_train, y_train)
y_pred = logistic_r.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Evaluating Logistic Regression')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print()

#-----------Softmax Regression------------
softmax_r = LogisticRegression(
    #multi_class='multinomial',   
    solver='lbfgs',              
    max_iter=1000,               
    C=1.0,                       
    random_state=42                 
)
softmax_r.fit(train_images, train_labels)
predict_labels = softmax_r.predict(test_images)

precision = precision_score(test_labels, predict_labels, average='macro', zero_division=0)
recall = recall_score(test_labels, predict_labels, average='macro', zero_division=0)
f1 = f1_score(test_labels, predict_labels, average='macro', zero_division=0)

print('Evaluating Softmax Regression')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")