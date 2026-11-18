import torch
import torch.nn as nn
import torch.nn.functional as F

class AMDM(nn.Module):
    def __init__(self, config, freq_matrix):
        super(AMDM, self).__init__()
        self.config = config
        
        # 1. Register Transition Frequency Matrix for Multi-Hop
        self.register_buffer('freq', freq_matrix)
        
        # 2. Embeddings
        self.loc_emb = nn.Embedding(config.num_regions, config.d_model)
        self.poi_proj = nn.Linear(config.num_poi, config.d_model)
        
        # 3. Spatial-Temporal Encoder (Local CNN)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * config.grid_size * config.grid_size, config.d_model)
        )
        
        # 4. Attention Aggregation Decoder
        self.attn = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=config.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(config.d_model)
        
        # 5. Fusion & Prediction Layer
        self.fusion_layer = nn.Linear(config.d_model * 2, config.d_model)
        self.output_head = nn.Linear(config.d_model, config.num_regions)
        self.dropout = nn.Dropout(config.dropout)

    def multi_hop_score(self, origin, candidate):
        """
        Calculates score based on 1-hop, 2-hop, and 3-hop paths.
        """
        P = self.freq / self.freq.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        direct = P[origin, candidate]
        two_hop = torch.einsum("bi,ij,bj->b", P[origin], P, P[:, candidate])
        three_hop = torch.einsum("bi,ij,jk,bk->b", P[origin], P, P, P[:, candidate])
        
        return direct + two_hop + three_hop

    def forward(self, batch):
        hist_origins = batch['hist_origins']
        hist_dests = batch['hist_dests']
        current_origin = batch['current_origin']
        local_grid = batch['local_grid']
        poi = batch['poi']
        
        B, L = hist_origins.size()
        
        # --- 1. Multi-Dimensional Embeddings ---
        o_emb = self.loc_emb(hist_origins)
        d_emb = self.loc_emb(hist_dests)
        poi_emb = self.poi_proj(poi)
        
        # --- 2. Spatial-Temporal Encoding ---
        spatial_feat = self.local_cnn(local_grid).unsqueeze(1)
        
        # --- 3. Attention Aggregation Decoder ---
        kv = o_emb + d_emb + poi_emb
        query = self.loc_emb(current_origin).unsqueeze(1)
        
        attn_out, _ = self.attn(query, kv, kv)
        z = self.norm(attn_out.squeeze(1) + query.squeeze(1))
        
        # --- 4. Multi-Hop Transition Reconstruction ---
        candidate_regions = torch.arange(self.config.num_regions, device=z.device).unsqueeze(0).expand(B, -1)
        origin_expanded = current_origin.unsqueeze(1).expand(-1, self.config.num_regions)
        
        recon_scores = self.multi_hop_score(origin_expanded.reshape(-1), candidate_regions.reshape(-1))
        recon_scores = recon_scores.view(B, self.config.num_regions)
        
        # --- 5. Fusion and Prediction ---
        z_expanded = z.unsqueeze(1).expand(-1, self.config.num_regions, -1)
        recon_expanded = recon_scores.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        
        fused = torch.cat([z_expanded, recon_expanded], dim=-1)
        fused = self.dropout(F.relu(self.fusion_layer(fused)))
        
        logits = self.output_head(fused)
        logits = logits.mean(dim=-1)
        
        return logits
