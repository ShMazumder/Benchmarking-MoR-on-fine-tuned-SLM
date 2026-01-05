"""Data package initialization"""
from .shakespeare import get_shakespeare_loaders, TinyShakespeareDataset
from .wikitext import get_wikitext_loaders, WikiText2Dataset

__all__ = [
    'get_shakespeare_loaders',
    'get_wikitext_loaders',
    'TinyShakespeareDataset',
    'WikiText2Dataset'
]
