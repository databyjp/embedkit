# ./src/embedkit/providers/colpali.py
"""ColPali embedding provider."""

from typing import Union, List, Optional
from pathlib import Path
import logging
import numpy as np
import torch
from PIL import Image

from ..utils import image_to_base64
from ..base import EmbeddingProvider, EmbeddingError, EmbeddingResponse

logger = logging.getLogger(__name__)


class ColPaliProvider(EmbeddingProvider):
    """ColPali embedding provider for document understanding."""

    def __init__(
        self,
        model_name: str,
        text_batch_size: int,
        image_batch_size: int,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            text_batch_size=text_batch_size,
            image_batch_size=image_batch_size,
            provider_name="ColPali",
        )

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                if self.model_name == "vidore/colpali-v1.3":
                    from colpali_engine.models import ColPali, ColPaliProcessor

                    self._model = ColPali.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=self.device,
                    ).eval()

                    self._processor = ColPaliProcessor.from_pretrained(self.model_name)

                elif self.model_name == "vidore/colSmol-256M":
                    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

                    self._model = ColIdefics3.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.bfloat16,
                            device_map=self.device,
                        ).eval()
                    self._processor = ColIdefics3Processor.from_pretrained(self.model_name)

                logger.info(f"Loaded {self.model_name} on {self.device}")

            except ImportError as e:
                raise EmbeddingError(
                    "ColPali not installed. Run: pip install colpali-engine"
                ) from e
            except Exception as e:
                raise EmbeddingError(f"Failed to load model: {e}") from e

    def embed_text(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """Generate embeddings for text inputs."""
        self._load_model()
        texts = self._normalize_text_input(texts)

        try:
            # Process texts in batches
            all_embeddings: List[np.ndarray] = []

            for i in range(0, len(texts), self.text_batch_size):
                batch_texts = texts[i : i + self.text_batch_size]
                processed = self._processor.process_queries(batch_texts).to(self.device)

                with torch.no_grad():
                    batch_embeddings = self._model(**processed)
                    all_embeddings.append(batch_embeddings.cpu().float().numpy())

            # Concatenate all batch embeddings
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            return self._create_text_response(final_embeddings)

        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    def embed_image(
        self, images: Union[Path, str, List[Union[Path, str]]]
    ) -> EmbeddingResponse:
        """Generate embeddings for images."""
        self._load_model()
        images = self._normalize_image_input(images)

        try:
            # Process images in batches
            all_embeddings: List[np.ndarray] = []
            all_b64_data: List[str] = []
            all_content_types: List[str] = []

            for i in range(0, len(images), self.image_batch_size):
                batch_images = images[i : i + self.image_batch_size]
                pil_images = []
                batch_b64_data = []
                batch_content_types = []

                for img_path in batch_images:
                    if not img_path.exists():
                        raise EmbeddingError(f"Image not found: {img_path}")

                    with Image.open(img_path) as img:
                        pil_images.append(img.convert("RGB"))
                    b64, content_type = image_to_base64(img_path)
                    batch_b64_data.append(b64)
                    batch_content_types.append(content_type)

                processed = self._processor.process_images(pil_images).to(self.device)

                with torch.no_grad():
                    batch_embeddings = self._model(**processed)
                    all_embeddings.append(batch_embeddings.cpu().float().numpy())
                    all_b64_data.extend(batch_b64_data)
                    all_content_types.extend(batch_content_types)

            # Concatenate all batch embeddings
            final_embeddings = np.concatenate(all_embeddings, axis=0)
            return self._create_image_response(
                final_embeddings, all_b64_data, all_content_types
            )

        except Exception as e:
            raise EmbeddingError(f"Failed to embed images: {e}") from e
