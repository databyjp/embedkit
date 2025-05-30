"""Base classes for EmbedKit."""

from abc import ABC, abstractmethod
from typing import Union, List
from pathlib import Path
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text objects."""
        pass

    @abstractmethod
    def embed_image(
        self, images: Union[Path, str, List[Union[Path, str]]]
    ) -> np.ndarray:
        """Generate embeddings for images."""
        pass


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""

    pass
