"""Data package initialization"""
from .shakespeare import get_shakespeare_loaders, TinyShakespeareDataset
from .wikitext import get_wikitext_loaders, WikiText2Dataset
from .bangla import get_bangla_loaders, BanglaSLMDataset

__all__ = [
    'get_shakespeare_loaders',
    'get_wikitext_loaders',
    'get_bangla_loaders',
    'TinyShakespeareDataset',
    'WikiText2Dataset',
    'BanglaSLMDataset'
]
