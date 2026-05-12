from scr.random_forest import RandomForest
from scr.data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

dataset = load_data()
X, y = dataset[:, 0:-1], dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = RandomForest(
    n_trees=10,
    max_depth=6,
    min_samples=5
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Random Forest - scratch f1 score: {f1:.6f}')