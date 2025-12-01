import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os

class ContextualFusionTransformer(nn.Module):
    """
    Transformer-based model for Multi-Modal Fusion.
    
    Inputs:
    - Text Embeddings (Batch, Seq, 384)
    - Audio/Tonal Embeddings (Batch, Seq, 64)
    - Role Embeddings (Batch, Seq, 384)
    
    Output:
    - Importance Score (Batch, Seq, 1)
    """
    def __init__(
        self,
        text_dim: int = 384,
        audio_dim: int = 64,
        role_dim: int = 384,
        model_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Feature Projections
        self.text_proj = nn.Linear(text_dim, model_dim)
        self.audio_proj = nn.Linear(audio_dim, model_dim)
        self.role_proj = nn.Linear(role_dim, model_dim)
        
        # 2. Fusion (Concatenation -> Projection)
        # We project each modality to model_dim, then sum them (additive fusion)
        # Alternatively, we could concat (3 * model_dim) -> model_dim
        self.fusion_norm = nn.LayerNorm(model_dim)
        
        # 3. Transformer Encoder (Contextual Understanding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head
        self.head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        text_emb: torch.Tensor,
        audio_emb: Optional[torch.Tensor] = None,
        role_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_emb: (Batch, Seq, text_dim)
            audio_emb: (Batch, Seq, audio_dim) - Optional, defaults to zeros
            role_emb: (Batch, Seq, role_dim) - Optional, defaults to zeros
            mask: (Batch, Seq) - Padding mask (True for padding positions)
            
        Returns:
            scores: (Batch, Seq, 1)
        """
        batch_size, seq_len, _ = text_emb.shape
        device = text_emb.device
        
        # Project Text
        x = self.text_proj(text_emb)
        
        # Add Audio (if present)
        if audio_emb is not None:
            # Handle dimension mismatch if necessary (though we expect correct dim)
            if audio_emb.shape[-1] != self.audio_proj.in_features:
                # Placeholder for safety, though caller should ensure dim
                pass
            x = x + self.audio_proj(audio_emb)
            
        # Add Role (if present)
        if role_emb is not None:
            x = x + self.role_proj(role_emb)
            
        # Normalize
        x = self.fusion_norm(x)
        
        # Transformer
        # src_key_padding_mask expects True for padded positions
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Output Head
        scores = self.head(x)
        
        return scores

def save_model(model, path="fusion_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path="fusion_model.pth", device="cpu"):
    model = ContextualFusionTransformer()
    if os.path.exists(path):
        if torch.cuda.is_available() and device == "cuda":
            model.load_state_dict(torch.load(path))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
