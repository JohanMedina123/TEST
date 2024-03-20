from model_convolucional import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.optim as optim

class LoadData :

    # Descarga y prepara el conjunto de datos EMNIST
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, transform=transform, download=True)
    test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Instancia la red y el optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=47).to(device)  # Cambiar el número de clases a 47 para EMNIST
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)