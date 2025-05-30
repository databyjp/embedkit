from embedkit import EmbedKit
from embedkit.models import Model
from pathlib import Path
import os


def get_online_image(url: str) -> Path:
    """Download an image from a URL and return its local path."""
    import requests
    from tempfile import NamedTemporaryFile

    response = requests.get(url)
    response.raise_for_status()

    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(response.content)
    temp_file.close()

    return Path(temp_file.name)


def get_sample_image() -> Path:
    """Get a sample image for testing."""
    url = "https://upload.wikimedia.org/wikipedia/commons/b/b8/English_Wikipedia_HomePage_2001-12-20.png"
    return get_online_image(url)


# kit = EmbedKit.colpali(model=Model.COLPALI_V1_3)

# embeddings = kit.embed_text("Hello world")
# assert embeddings.shape[0] == 1
# assert len(embeddings.shape) == 3

# embeddings = kit.embed_image(get_sample_image())
# assert embeddings.shape[0] == 1
# assert len(embeddings.shape) == 3


kit = EmbedKit.cohere(model=Model.COHERE_V4_0, api_key=os.getenv("COHERE_API_KEY"))

embeddings = kit.embed_text("Hello world")
assert embeddings.shape[0] == 1
assert len(embeddings.shape) == 2

embeddings = kit.embed_image(get_sample_image())
assert embeddings.shape[0] == 1
assert len(embeddings.shape) == 2
