#Data Format
#Format your CSV file with the following columns:
#origin: (int) Origin grid ID
#destination: (int) Destination grid ID
#timestamp: (string) e.g., "2023-10-25 14:30:00"
#user_id: (int) Unique user ID
#poi: (list/array) POI features (optional)


python train.py --data_path data/train.csv --epochs 200 --batch_size 64

python evaluate.py --model_path saved_models/amdm_best.pth --data_path data/test.csv


---

### 2. `requirements.txt`
```text
### torch>=2.0.0
### numpy>=1.21.0
###pandas>=1.3.0
###scikit-learn>=1.0.0
###tqdm>=4.62.0

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import copy

from config import get_config
from dataset import AMDDataset, build_transition_frequency
from model import AMDM

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        for k, v in batch.items():
            batch[k] = v.to(device)
            
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch['target'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            for k, v in batch.items():
                batch[k] = v.to(device)
            logits = model(batch)
            loss = criterion(logits, batch['target'])
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs('saved_models', exist_ok=True)
    
    # 1. Load Data
    print("Loading training dataset...")
    train_dataset = AMDDataset(config.data_path, config.seq_len, config.num_regions)
    # Assuming a 80/20 split for train/val if not explicitly provided
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 2. Build Frequency Matrix
    print("Building transition frequency matrix...")
    freq_matrix = build_transition_frequency(config.data_path, config.num_regions).to(device)
    
    # 3. Initialize Model & Optimizer
    model = AMDM(config, freq_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop with Early Stopping (200 Epochs)
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"Starting training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Check Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "saved_models/amdm_best.pth")
            print(f" --> Best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
    print("Training complete. Best Validation Loss: {:.4f}".format(best_val_loss))

if __name__ == "__main__":
    main()
