import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import get_config
from dataset import AMDDataset, build_transition_frequency
from model import AMDM
from metrics import calculate_metrics

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Test Data
    test_dataset = AMDDataset(config.data_path, config.seq_len, config.num_regions)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 2. Load Model
    freq_matrix = build_transition_frequency('data/train.csv', config.num_regions).to(device)
    model = AMDM(config, freq_matrix).to(device)
    
    # Load the weights saved from the training script
    model.load_state_dict(torch.load('saved_models/amdm_best.pth', map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # 3. Run Inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            for k, v in batch.items():
                batch[k] = v.to(device)
                
            logits = model(batch)
            all_preds.append(logits.cpu())
            all_targets.append(batch['target'].cpu())
            
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 4. Calculate Metrics
    # Note: For MDE, you need to pass the spatial distance matrix here
    results = calculate_metrics(all_preds, all_targets, distance_matrix=None)
    
    print("\n--- Final Evaluation Results ---")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
