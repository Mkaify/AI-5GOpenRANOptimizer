import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

class EnergyEfficiencyModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnergyEfficiencyModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        numeric_df = df.select_dtypes(include=[np.number])
        if 'id' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['id'])
        
        self.X = numeric_df.iloc[:, :-1].values.astype(np.float32)
        self.y = numeric_df.iloc[:, -1:].values.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader)

def main():
    # Setup data path
    data_path = "data/energy_efficiency/train.csv"
    train_dataset = SimpleDataset(data_path)
    
    # Dynamic hyperparameter allocation based on columns
    input_size = train_dataset.X.shape[1]
    hidden_size = 64
    output_size = 1
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 16

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnergyEfficiencyModel(input_size, hidden_size, output_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print(f"Starting Energy Efficiency model training (Input size: {input_size})...")
    for epoch in range(1, num_epochs + 1):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch: [{epoch}/{num_epochs}]\tTrain Loss: {loss:.4f}")

    # Save model weights safely
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/energy_efficiency.pt")
    print("Model successfully saved to models/energy_efficiency.pt")

if __name__ == '__main__':
    main()