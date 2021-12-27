__all__ = ['CategoryEncoder', 'MultiCategoryEncoder']

from tai_chi_tuna.front.typer import LIST
from .basic import EntryModel

from torch import nn

class CategoryEncoder(EntryModel):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.model = nn.Embedding(
            num_embeddings,
            embedding_dim)
        
    def forward(self, idx):
        return self.model(idx)
    
    @classmethod
    def from_quantify(
        cls,
        quantify,
        embedding_dim: LIST(
            options=[4, 8, 16, 32, 64, 128, 256, 512], default=128) = 128):
        num_embeddings = len(quantify.category)
        obj = cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        obj.out_features = embedding_dim
        return obj
    
class MultiCategoryEncoder(EntryModel):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.model = nn.Embedding(
            num_embeddings,
            embedding_dim)
        
    def forward(self, idx):
        return idx.float()@self.model.weight
    
    @classmethod
    def from_quantify(
        cls,
        quantify,
        embedding_dim: LIST(
            options=[4, 8, 16, 32, 64, 128, 256, 512], default=128) = 128):
        num_embeddings = len(quantify.category)
        obj = MultiCategoryEncoder(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        obj.out_features = embedding_dim
        return obj
