#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

//g++ -O2 -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` svm.cpp -o svm_cpp`python3.12-config --extension-suffix`

namespace py = pybind11;

// Función para calcular la predicción basada en el hiperplano de decisión
std::vector<int> svm_predict(const std::vector<std::vector<double>>& X, 
                             const std::vector<double>& weights, 
                             double bias) {
    std::vector<int> predictions;
    for (const auto& point : X) {
        double decision_value = bias;
        for (size_t i = 0; i < point.size(); ++i) {
            decision_value += weights[i] * point[i];
        }
        // Predicción basada en el signo de la función de decisión
        predictions.push_back(decision_value >= 0 ? 1 : -1);
    }
    return predictions;
}

// Exponer la función a Python
PYBIND11_MODULE(svm_cpp, m) {
    m.def("svm_predict", &svm_predict, "SVM Prediction using decision hyperplane",
          py::arg("X"), py::arg("weights"), py::arg("bias"));
}
