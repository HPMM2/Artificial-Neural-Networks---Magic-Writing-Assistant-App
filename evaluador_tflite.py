# evaluador_tflite.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import tensorflow as tf  # para TFLite

# Paso 1: Transformaciones (igual que antes)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Paso 2: Cargar dataset y aplicar transformaciones con TorchDataset
print("Cargando y transformando conjunto de prueba...")
dataset = load_dataset("pittawat/letter_recognition")

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

test_dataset = TorchDataset(dataset['test'], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1)  # batch_size=1 para TFLite

# Paso 3: Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="letter_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Verificación forma esperada
print("Input shape esperado por el modelo:", input_details[0]['shape'])

# Paso 4: Evaluar el modelo TFLite
correct = 0
total = 0

for images, labels in test_loader:
    # Convertir imagen a numpy float32
    input_data = images.numpy().astype(np.float32)  # shape: [1, 1, 28, 28]

    # Reordenar a [1, 28, 28, 1] para TFLite (batch, height, width, channels)
    input_data = np.transpose(input_data, (0, 2, 3, 1))

    # Asegurar que input_data coincide con la forma esperada
    expected_shape = input_details[0]['shape']
    if input_data.shape != tuple(expected_shape):
        input_data = np.reshape(input_data, expected_shape)

    # Pasar input al intérprete
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Obtener resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted = np.argmax(output_data, axis=1)

    total += 1
    correct += (predicted[0] == labels.item())

accuracy = 100 * correct / total
print(f"Precisión del modelo TFLite en el conjunto de prueba: {accuracy:.2f}%")
