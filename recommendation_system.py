from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Datos de ejemplo: [precio, valoración]
X = [[10, 4], [20, 5], [15, 3], [30, 5], [25, 4], [12, 2], [40, 5], [35, 4]]
y = [0, 1, 0, 1, 1, 0, 1, 1]  # 0 = no recomendado, 1 = recomendado

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
print("Precisión:", accuracy_score(y_test, predicciones))

# Recomendar un producto nuevo
nuevo_producto = [[22, 4]]
print("Recomendación:", modelo.predict(nuevo_producto))
