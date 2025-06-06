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
)

# Get document embeddings
result = kit.embed_document("Hello world")
print(result.objects[0].embedding.shape)  # 1D array

# Get query embeddings (for providers that support it)
result = kit.embed_query("Hello world")
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
)

# Different embeddings for queries vs documents
query_result = kit.embed_query("What is the capital of France?")
doc_result = kit.embed_document("Paris is the capital of France.")
```

### Snowflake
```python
kit = EmbedKit.snowflake(
    model=Model.Snowflake.ARCTIC_EMBED_L_V2_0,  # or ARCTIC_EMBED_M_V1_5
)

# Different embeddings for queries vs documents
query_result = kit.embed_query("What is the capital of France?")
doc_result = kit.embed_document("Paris is the capital of France.")
```

### Qwen
```python
# Lightweight model (0.6B parameters)
kit = EmbedKit.qwen(
    model=Model.Qwen.QWEN3_EMBEDDING_0_6B,
)

# Larger models (require more memory)
# kit = EmbedKit.qwen(
#     model=Model.Qwen.QWEN3_EMBEDDING_4B,
# )
# kit = EmbedKit.qwen(
#     model=Model.Qwen.QWEN3_EMBEDDING_8B,
# )

# Different embeddings for queries vs documents
query_result = kit.embed_query("What is the capital of France?")
doc_result = kit.embed_document("Paris is the capital of France.")
```

### ColPali
```python
kit = EmbedKit.colpali(
    model=Model.ColPali.COLPALI_V1_3,  # or COLSMOL_256M, COLSMOL_500M
)

# Same embeddings for queries and documents
query_result = kit.embed_query("What is the capital of France?")
doc_result = kit.embed_document("Paris is the capital of France.")
assert np.array_equal(query_result.objects[0].embedding, doc_result.objects[0].embedding)
```

### Jina
```python
kit = EmbedKit.jina(
    model=Model.Jina.CLIP_V2,
    api_key="your-api-key",
)

# Same embeddings for queries and documents
query_result = kit.embed_query("What is the capital of France?")
doc_result = kit.embed_document("Paris is the capital of France.")
assert np.array_equal(query_result.objects[0].embedding, doc_result.objects[0].embedding)
```

## Response Format

```python
class EmbeddingResponse:
    model_name: str
    model_provider: str
    input_type: str  # "text", "search_query", "search_document", "query", "image"
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
pytest -m qwen      # Run only Qwen tests

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
