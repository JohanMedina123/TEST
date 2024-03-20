from model_convolucional import *
from data_processing import *
import torch
import os

class Trainer:
    # Instancia la red y el optimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=47).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'saved_models', 'model_trained.pth')

#Entrenamiento de la red
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(LoadData.train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(LoadData.train_loader)}, Loss: {loss.item()}')

    # Evaluación del modelo
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in LoadData.test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy * 100:.2f}%')
    torch.save(model.state_dict(), model_path)
    print("NN Model trained and saved successfully.")