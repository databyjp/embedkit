# ./src/embedkit/models.py
"""Model definitions and enum for EmbedKit."""

from enum import Enum


class Model(Enum):
    """Available embedding models."""

    COLPALI_V1_3 = "colpali-v1.3"
    COHERE_V4_0 = "embed-v4.0"
