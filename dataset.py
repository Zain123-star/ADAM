import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AMDDataset(Dataset):
    def __init__(self, csv_file, seq_len=8, num_regions=1024, num_poi=10):
        self.data = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.num_regions = num_regions
        
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        
        self.user_groups = self.data.groupby('user_id')
        self.valid_users = [uid for uid, group in self.user_groups if len(group) > self.seq_len]
        
    def __len__(self):
        return len(self.valid_users)
    
    def __getitem__(self, idx):
        user_id = self.valid_users[idx]
        user_data = self.user_groups.get_group(user_id)
        
        max_start = len(user_data) - self.seq_len - 1
        start_idx = np.random.randint(0, max_start + 1)
        seq_data = user_data.iloc[start_idx : start_idx + self.seq_len + 1]
        
        origins = seq_data['origin'].values
        dests = seq_data['destination'].values
        
        # Input sequence
        hist_origins = torch.tensor(origins[:-1], dtype=torch.long)
        hist_dests = torch.tensor(dests[:-1], dtype=torch.long)
        
        # Target
        target = torch.tensor(dests[-1], dtype=torch.long)
        current_origin = torch.tensor(origins[-1], dtype=torch.long)
        
        # Dummy data for Local Grid and POI (Replace with real data if available)
        local_grid = torch.randn(1, 7, 7)
        poi_features = torch.randn(self.seq_len, 10)
        
        return {
            'hist_origins': hist_origins,
            'hist_dests': hist_dests,
            'current_origin': current_origin,
            'target': target,
            'local_grid': local_grid,
            'poi': poi_features
        }

def build_transition_frequency(csv_file, num_regions):
    """Builds the frequency matrix for multi-hop reconstruction."""
    df = pd.read_csv(csv_file)
    freq_matrix = torch.zeros(num_regions, num_regions)
    
    for _, row in df.iterrows():
        o, d = int(row['origin']), int(row['destination'])
        freq_matrix[o, d] += 1.0
        
    return freq_matrix
