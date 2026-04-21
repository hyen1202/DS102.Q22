from sklearn.svm import SVC
from data_loader import load_data
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train, y_train = load_data('chest_xray/train')
X_train = scaler.fit_transform(X_train)
X_test, y_test = load_data('chest_xray/test')
X_test = scaler.transform(X_test)

model = SVC(
    C = 1,
    kernel='linear'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')


'''
Nhận xét:
    1. Báo các chỉ số
    - Kết quả khi chạy SVC trong thư viện sklearn
        Precision: 0.7127
        Recall: 0.9923
        F1: 0.8296
    - Kết quả của SVM tự cài đặt (epoches = 200, lr = 0.0001, C = 0.1)
        Precision: 0.7314
        Recall: 0.9846
        F1: 0.8393
    2. Phân tích chỉ số
        - Recall cao chứng tỏ model ít bỏ sót ca PNEUMONIA
        - Precision thấp cho thấy model dự đoán PNEUMONIA quá nhiều
          -> model bị bias về phía PNEUMONIA 
    3. So sánh 
        - Cả hai đều có pattern giống nhau (Recall cao, Precision thấp) nhưng SVM tự code nhỉnh hơn nhưng không chênh lệch nhiều 
        - SVC không thay đổi khi đổi C từ 100 → 0.01, còn SVM tự code có thay đổi.
    4. Giải thích nguyên nhân
        - Recall cao hơn là do hai lớp mất cân bằng, số lượng PNEUMONIA trong train nhiều hơn NORMAL
        - SVC không thay đổi khi đổi C vì sau StandardScaler 
          dữ liệu đã được scale tốt, model ít nhạy cảm với C hơn,
          trong khi SVM tự code dùng SGD vẫn nhạy cảm với hyperparameter

'''