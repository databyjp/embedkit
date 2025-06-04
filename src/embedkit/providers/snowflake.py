"""Snowflake Arctic Embed provider."""

from typing import Union, List
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path

from ..models import Model
from ..base import EmbeddingProvider, EmbeddingError, EmbeddingResponse

logger = logging.getLogger(__name__)


class SnowflakeProvider(EmbeddingProvider):
    """Snowflake Arctic Embed provider for text embeddings."""

    def __init__(
        self,
        model: Model.Snowflake,
        text_batch_size: int,
        image_batch_size: int,
        device: str = None,
    ):
        super().__init__(
            model_name=model.value,
            text_batch_size=text_batch_size,
            image_batch_size=image_batch_size,
            provider_name="Snowflake",
        )
        self._device = device
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
                if self._device:
                    self._model = self._model.to(self._device)
                logger.info(f"Loaded {self.model_name} on {self._device or 'default device'}")
            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                ) from e
            except Exception as e:
                raise EmbeddingError(f"Failed to load model: {e}") from e

    def embed_text(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """Generate text embeddings using the Snowflake Arctic Embed model."""
        self._load_model()
        texts = self._normalize_text_input(texts)

        try:
            all_embeddings = []

            # Process texts in batches
            for i in range(0, len(texts), self.text_batch_size):
                batch_texts = texts[i : i + self.text_batch_size]
                # Use query prompt for all inputs since we don't distinguish between queries and documents
                batch_embeddings = self._model.encode(
                    batch_texts,
                    prompt_name="query",
                    convert_to_numpy=True,
                )
                all_embeddings.append(batch_embeddings)

            # Concatenate all batch embeddings
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            return self._create_text_response(final_embeddings)

        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    def embed_image(
        self,
        images: Union[Path, str, List[Union[Path, str]]],
    ) -> EmbeddingResponse:
        """Raise error as Snowflake Arctic Embed does not support image embeddings."""
        raise EmbeddingError(
            "Snowflake Arctic Embed does not support image embeddings"
        )
