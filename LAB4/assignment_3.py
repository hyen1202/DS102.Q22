from sklearn.tree import DecisionTreeClassifier
from scr.data_loader import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


dataset = load_data()
X, y = dataset[:, 0:-1], dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

dt_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()

dt_model.fit(X_train, y_train)
rfc_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
rfc_pred = rfc_model.predict(X_test)

f1_dt = f1_score(y_test, dt_pred)
f1_rfc = f1_score(y_test, dt_pred)

print(f'Decision tree by sklearn: f1 = {f1_dt:.6f}')
print(f'Random forest by sklearn: f1 = {f1_rfc:.6f}')