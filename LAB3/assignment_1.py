from data_loader import load_data
from svm import SVM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train, y_train = load_data('chest_xray/train')
X_train = scaler.fit_transform(X_train)
X_test, y_test = load_data('chest_xray/test')
X_test = scaler.transform(X_test)

model = SVM(
    C = 0.1,
    epoches = 200,
    lr = 0.0001
)

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

metrics = model.evaluate(y_test, y_predict)
print(f'Precision: {metrics['precision']:.4f}')
print(f'Recall: {metrics['recall']:.4f}')
print(f'F1: {metrics['f1']:.4f}')

plt.plot(model.loss_list)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Training loss')
plt.show()


