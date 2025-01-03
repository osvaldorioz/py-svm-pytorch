from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
import svm_cpp

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

# Generar datos sintéticos para SVM
def generate_data(num_samples=200):
    torch.manual_seed(42)
    X = torch.rand((num_samples, 2)) * 2 - 1  # Rango [-1, 1]
    y = torch.where(X[:, 0] * 2 + X[:, 1] - 0.5 > 0, 1, -1)  # Hiperplano: 2x + y - 0.5 = 0
    return X, y

# Generar datos
X, y = generate_data(num_samples=300)

# Dividir los datos en entrenamiento y prueba
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Entrenar el modelo SVM (usando gradiente descendente básico en PyTorch)
class SVMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(2))
        self.bias = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, X):
        return X @ self.weights + self.bias
    

@app.post("/svm")
def calculo(epochs: int):
    # Definir el modelo y el optimizador
    model = SVMModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.HingeEmbeddingLoss()

    # Entrenamiento
    #epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs * y_train, torch.ones_like(y_train))
        loss.backward()
        optimizer.step()

    # Pesos entrenados
    weights = model.weights.detach().numpy().tolist()
    bias = model.bias.item()

    # Predicción en C++
    test_data = X_test.numpy().tolist()
    predictions = svm_cpp.svm_predict(test_data, weights, bias)

    # Evaluar precisión
    accuracy = sum([1 if pred == true else 0 for pred, true in zip(predictions, y_test.numpy())]) / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Visualización
    df_train = pd.DataFrame(X_train.numpy(), columns=["x1", "x2"])
    df_train['label'] = y_train.numpy()

    df_test = pd.DataFrame(X_test.numpy(), columns=["x1", "x2"])
    df_test['predicted'] = predictions

    plt.figure(figsize=(10, 6))
    plt.scatter(df_train['x1'], df_train['x2'], c=df_train['label'], cmap='viridis', label='Training Data')
    plt.scatter(df_test['x1'], df_test['x2'], c=df_test['predicted'], cmap='plasma', marker='x', label='Test Predictions')
    plt.axline((0, 0.5), slope=-2, color='red', linestyle='--', label='True Hyperplane')  # Hiperplano real
    plt.legend()
    plt.title("Resultados de Clasificación por SVM")
    output_file = 'svm_results.png'
    plt.savefig(output_file)
    plt.close()
    # Regresar el archivo como respuesta
    return FileResponse(output_file, media_type="image/png", filename=output_file)



    