import torch
import numpy as np
from sklearn.metrics import ndcg_score

def calculate_metrics(predictions, targets, distance_matrix=None):
    """
    Calculates MAP, MRR, NDCG@10, Recall@k, and MDE.
    """
    N = targets.size(0)
    
    # Sort predictions descending
    _, sorted_indices = torch.sort(predictions, dim=1, descending=True)
    
    # 1. MRR (Mean Reciprocal Rank)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = (1.0 / ranks.float()).mean().item()
    
    # 2. NDCG@10
    y_true = np.zeros_like(predictions.cpu().numpy())
    y_true[np.arange(N), targets.cpu().numpy()] = 1
    ndcg = ndcg_score(y_true, predictions.cpu().numpy(), k=10)
    
    # 3. MAP (Simplified for single-label tasks)
    map_score = mrr 
    
    # 4. Recall@1, Recall@5, Recall@10
    recall_1 = (ranks <= 1).float().mean().item()
    recall_5 = (ranks <= 5).float().mean().item()
    recall_10 = (ranks <= 10).float().mean().item()
    
    # 5. MDE (Mean Distance Error)
    mde = 0.0
    if distance_matrix is not None:
        top1_preds = sorted_indices[:, 0]
        mde = distance_matrix[top1_preds, targets].mean().item()
        
    return {
        'MAP': map_score,
        'MRR': mrr,
        'NDCG@10': ndcg,
        'MR@1': recall_1,
        'MR@5': recall_5,
        'MR@10': recall_10,
        'MDE': mde
    }
