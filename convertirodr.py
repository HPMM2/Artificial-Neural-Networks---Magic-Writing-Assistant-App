import torch
import torch.nn as nn
import torch.nn.functional as F

# Modelo igual al que usaste
class LetterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Cargar modelo y estado
model = LetterClassifier()
model.load_state_dict(torch.load("letter_model.pt"))
model.eval()

# Dummy input (batch=1, canal=1, alto=28, ancho=28)
dummy_input = torch.randn(1, 1, 28, 28)

# Exportar a ONNX
onnx_path = "letter_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path,
                  input_names=['input'], output_names=['output'],
                  opset_version=11)

print(f"Modelo exportado a ONNX en {onnx_path}")

# Convertir ONNX a TensorFlow y luego a TFLite
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Cargar modelo ONNX
onnx_model = onnx.load(onnx_path)

# Preparar modelo TF
tf_rep = prepare(onnx_model)

# Exportar modelo TF (SavedModel)
tf_model_path = "./letter_tf_model"
tf_rep.export_graph(tf_model_path)
print(f"Modelo TensorFlow exportado en {tf_model_path}")

# Convertir SavedModel TF a TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Guardar archivo TFLite
tflite_path = "letter_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Modelo convertido a TFLite y guardado en {tflite_path}")
