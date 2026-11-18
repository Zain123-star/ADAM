import argparse

def get_config():
    parser = argparse.ArgumentParser(description="AMDM Configuration")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/train.csv', help='Path to dataset')
    parser.add_argument('--seq_len', type=int, default=8, help='Historical OD sequence length')
    parser.add_argument('--num_regions', type=int, default=1024, help='Total number of grid regions')
    parser.add_argument('--num_poi', type=int, default=10, help='Number of POI categories')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256, help='Embedding dimension (H_dm)')
    parser.add_argument('--grid_size', type=int, default=7, help='Local grid size (I x I)')
    parser.add_argument('--num_heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--max_hops', type=int, default=3, help='Max multi-hop path length (K)')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    return parser.parse_args()
