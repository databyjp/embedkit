# EmbedKit

A Python library for generating embeddings from text and images using various models (Cohere, ColPali).

## Usage

See [main.py](main.py) for examples.

```python
from embedkit import EmbedKit
from embedkit.models import Model

# Using ColPali
kit = EmbedKit.colpali(model=Model.COLPALI_V1_3)
embeddings = kit.embed_text("Hello world")
embeddings = kit.embed_image("path/to/image.png")

# Using Cohere
kit = EmbedKit.cohere(
    model=Model.COHERE_V4_0,
    api_key="your_api_key",
    text_input_type=CohereInputType.SEARCH_DOCUMENT,
)
embeddings = kit.embed_text("Hello world")
embeddings = kit.embed_image("path/to/image.png")
```

## License

MIT
