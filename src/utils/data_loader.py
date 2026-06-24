import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PNPDataset(Dataset):
    def __init__(self, data_file: str):
        # Load data
        raw_data = pd.read_csv(data_file)
        
        # Automatically extract only numerical features (float/int) for the neural network
        # This completely filters out 'timestamp', 'Band', and other text string types
        self.data = raw_data.select_dtypes(include=[np.number])
        
        # Exclude tracking identifiers from features if present
        if 'id' in self.data.columns:
            self.data = self.data.drop(columns=['id'])
            
        # Split into input features and target output using positional alignment
        self.input_data = self.data.iloc[:, :-1].values.astype(np.float32)
        self.output_data = self.data.iloc[:, -1:].values.astype(np.float32)
        
        self.input_dim = self.input_data.shape[1]
        self.output_dim = self.output_data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.Tensor(self.input_data[idx])
        targets = torch.Tensor(self.output_data[idx])
        return inputs, targets