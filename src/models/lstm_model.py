import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """Dataset PyTorch pour les séries temporelles"""
    
    def __init__(self, X, y, sequence_length=24):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length]
        )


class LSTMModel(nn.Module):
    """Architecture LSTM pour la prédiction de séries temporelles"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Prendre la dernière sortie
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output.squeeze()


class LSTMRegressor(BaseEstimator):
    """Wrapper sklearn pour le modèle LSTM"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, 
                 dropout=0.2, sequence_length=24, learning_rate=0.001,
                 epochs=50, batch_size=32, early_stopping_patience=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X, y, X_val=None, y_val=None):
        # Créer le modèle
        self.model = LSTMModel(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout
        ).to(self.device)
        
        # Optimiseur et perte
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Datasets et loaders
        train_dataset = TimeSeriesDataset(X, y, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Entraînement
        self.model.train()
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Sauvegarder le meilleur modèle
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        self.model.load_state_dict(best_model_state)
                        break
                
                self.model.train()
        
        return self
    
    def predict(self, X):
        self.model.eval()
        predictions = []
        
        # Créer des séquences
        for i in range(len(X) - self.sequence_length):
            seq = torch.FloatTensor(X.iloc[i:i + self.sequence_length].values).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(seq)
                predictions.append(pred.item())
        
        # Pour les sequence_length premières valeurs, utiliser la dernière prédiction disponible
        result = np.full(len(X), np.nan)
        result[self.sequence_length:] = predictions
        
        # Forward fill pour les premières valeurs
        result = pd.Series(result).fillna(method='ffill').fillna(method='bfill').values
        
        return result
