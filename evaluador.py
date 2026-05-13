# evaluador.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import torch.nn.functional as F
import torch.nn as nn

# Paso 1: Definir modelo (debe coincidir con el usado en entrenamiento)
class LetterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Paso 2: Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Paso 3: Cargar dataset y aplicar transformaciones con TorchDataset
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
test_loader = DataLoader(test_dataset, batch_size=64)

# Paso 4: Cargar modelo
model = LetterClassifier()
model.load_state_dict(torch.load("letter_model.pt"))
model.eval()

# Paso 5: Evaluar
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.2f}%")
