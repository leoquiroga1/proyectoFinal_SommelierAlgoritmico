import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Cargarmos del dataset local y utilizamos sep=';' porque el dataset que descargamos tiene separador punto y coma, igualmente decidimos iniciar con los vinos rojos.
data = pd.read_csv('winequality-red.csv', sep=';')

# 2. Definimos la etiqueta binaria de clasificación de los vinos (Premium >= 7, Estándar < 7)
data['Premium'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# 3. Separarmos las 11 variables de entrada (X) y la etiqueta (y)
X = data.drop(['quality', 'Premium'], axis=1)
y = data['Premium']

# 4. Hacemos un split estricto antes de la transformación (asi verificamos tener aparte los datos para test y entrenamiento).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Entrenarmos el modelo a traves del arbol de decision (Decision Tree Classifier) con profundidad máxima de 5 y balanceando las clases
modelo = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
modelo.fit(X_train, y_train)

# 6. Ejecutamos las predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# 7. Imprimimos las métricas que definimos inicialmente en la propuesta de proyecto (Accuracy, Precision, Recall y F1-Score)
print("--- MÉTRICAS DE EVALUACIÓN (Clase Positiva: Premium) ---")
print(f"Accuracy (Exactitud): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision (Precisión): {precision_score(y_test, y_pred):.2f}")
print(f"Recall (Exhaustividad): {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")

# 8. Exportamos el modelo para usarlo en la API
with open('wine_quality_model.pkl', 'wb') as f:
    pickle.dump(modelo, f)
print("\nModelo guardado exitosamente como 'wine_quality_model.pkl'")

# 9. Mostramos la gráfica de la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Estándar (0)', 'Premium (1)'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - Sommelier Algorítmico")
plt.show()