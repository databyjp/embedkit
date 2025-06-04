# EmbedKit

A unified interface for text and image embeddings, supporting multiple providers.

## Installation

```bash
pip install embedkit
```

## Quick Start

```python
from embedkit import EmbedKit
from embedkit.classes import Model, CohereInputType, SnowflakeInputType

# Initialize a provider
kit = EmbedKit.cohere(
    model=Model.Cohere.EMBED_V4_0,
    api_key="your-api-key",
    text_input_type=CohereInputType.SEARCH_QUERY,
)

# Get text embeddings
result = kit.embed_text("Hello world")
print(result.objects[0].embedding.shape)  # 1D array

# Get image embeddings
result = kit.embed_image("path/to/image.png")
print(result.objects[0].embedding.shape)  # 1D array
print(result.objects[0].source_b64)  # Base64 encoded image
```

## Supported Providers

### Cohere
```python
kit = EmbedKit.cohere(
    model=Model.Cohere.EMBED_V4_0,  # or EMBED_ENGLISH_V3_0, EMBED_MULTILINGUAL_V3_0, etc.
    api_key="your-api-key",
    text_input_type=CohereInputType.SEARCH_QUERY,  # or SEARCH_DOCUMENT
)
```

### Snowflake
```python
kit = EmbedKit.snowflake(
    model=Model.Snowflake.ARCTIC_EMBED_L_V2_0,  # or ARCTIC_EMBED_M_V1_5
    text_input_type=SnowflakeInputType.QUERY,  # or DOCUMENT
)
```

### ColPali
```python
kit = EmbedKit.colpali(
    model=Model.ColPali.COLPALI_V1_3,  # or COLSMOL_256M, COLSMOL_500M
)
```

### Jina
```python
kit = EmbedKit.jina(
    model=Model.Jina.CLIP_V2,
    api_key="your-api-key",
)
```

## Response Format

```python
class EmbeddingResponse:
    model_name: str
    model_provider: str
    input_type: str
    objects: List[EmbeddingObject]

class EmbeddingObject:
    embedding: np.ndarray  # 1D array for everything except ColPali
    source_b64: Optional[str]  # Base64 encoded source for images and PDFs
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests for specific providers
pytest -m cohere    # Run only Cohere tests
pytest -m colpali   # Run only ColPali tests
pytest -m jina      # Run only Jina tests
pytest -m snowflake # Run only Snowflake tests

# Additional options
pytest -v           # Verbose output
pytest -s           # Show print statements
pytest -x           # Stop on first failure
```

## Requirements

- Python 3.10+

## License

MIT

## GitHub

https://github.com/databyjp/embedkit
