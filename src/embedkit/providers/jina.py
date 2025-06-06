"""Jina embedding provider."""

from typing import Union, List, Dict, Any
from pathlib import Path
import numpy as np
import logging
import requests

from ..models import Model
from ..utils import image_to_base64
from ..base import EmbeddingProvider, EmbeddingError, EmbeddingResponse

logger = logging.getLogger(__name__)


class JinaProvider(EmbeddingProvider):
    """Jina embedding provider for text and image embeddings."""

    def __init__(
        self,
        api_key: str,
        model: Model.Jina,
        text_batch_size: int,
        image_batch_size: int,
    ):
        super().__init__(
            model_name=model.value,
            text_batch_size=text_batch_size,
            image_batch_size=image_batch_size,
            provider_name="Jina",
        )
        self.api_key = api_key
        self._base_url = "https://api.jina.ai/v1/embeddings"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _make_request(self, inputs: List[Dict[str, str]]) -> List[np.ndarray]:
        """Make a request to the Jina API."""
        try:
            response = requests.post(
                self._base_url,
                headers=self._headers,
                json={
                    "model": self.model_name,
                    "input": inputs,
                },
            )
            response.raise_for_status()
            data = response.json()
            return [np.array(item["embedding"]) for item in data["data"]]
        except requests.exceptions.RequestException as e:
            raise EmbeddingError(f"Failed to make request to Jina API: {e}") from e
        except (KeyError, ValueError) as e:
            raise EmbeddingError(f"Failed to parse Jina API response: {e}") from e

    def embed_document(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """Generate text embeddings using the Jina API."""
        texts = self._normalize_text_input(texts)

        try:
            all_embeddings = []

            # Process texts in batches
            for i in range(0, len(texts), self.text_batch_size):
                batch_texts = texts[i : i + self.text_batch_size]
                # Format each text input as a dict with "text" key
                inputs = [{"text": text} for text in batch_texts]
                batch_embeddings = self._make_request(inputs)
                all_embeddings.extend(batch_embeddings)

            return self._create_text_response(all_embeddings)

        except Exception as e:
            raise EmbeddingError(f"Failed to embed text with Jina: {e}") from e

    def embed_image(
        self,
        images: Union[Path, str, List[Union[Path, str]]],
    ) -> EmbeddingResponse:
        """Generate embeddings for images using Jina API."""
        images = self._normalize_image_input(images)
        total_images = len(images)
        logger.info(f"Starting to process {total_images} images")

        try:
            all_embeddings = []
            all_b64_images = []

            # Process images in batches
            for i in range(0, len(images), self.image_batch_size):
                batch_images = images[i : i + self.image_batch_size]
                logger.info(f"Processing batch {i//self.image_batch_size + 1} of {(total_images + self.image_batch_size - 1)//self.image_batch_size} ({len(batch_images)} images)")
                inputs = []

                for image in batch_images:
                    if not image.exists():
                        raise EmbeddingError(f"Image not found: {image}")
                    b64_data, content_type = image_to_base64(image)
                    # For Jina API, we need to send the base64 data directly without the data URI prefix
                    inputs.append({"image": b64_data})
                    all_b64_images.append((b64_data, content_type))

                batch_embeddings = self._make_request(inputs)
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Successfully processed all {total_images} images")
            return self._create_image_response(
                all_embeddings,
                [b64 for b64, _ in all_b64_images],
                [content_type for _, content_type in all_b64_images],
            )

        except Exception as e:
            logger.error(f"Failed to embed images: {e}")
            raise EmbeddingError(f"Failed to embed image with Jina: {e}") from e
