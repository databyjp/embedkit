# ./main.py
from embedkit import EmbedKit
from embedkit.classes import Model, CohereInputType
from pathlib import Path
import os


def get_online_image(url: str) -> Path:
    """Download an image from a URL and return its local path."""
    import requests
    from tempfile import NamedTemporaryFile

    # Add User-Agent header to comply with Wikipedia's policy
    headers = {"User-Agent": "EmbedKit-Example/1.0"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(response.content)
    temp_file.close()

    return Path(temp_file.name)


def get_sample_image() -> Path:
    """Get a sample image for testing."""
    url = "https://upload.wikimedia.org/wikipedia/commons/b/b8/English_Wikipedia_HomePage_2001-12-20.png"
    return get_online_image(url)


def test_provider(kit: EmbedKit, expected_dim: int):
    """Test a provider with various inputs."""
    # Test text embedding
    results = kit.embed_text("Hello world")
    assert len(results.objects) == 1
    assert len(results.objects[0].embedding.shape) == expected_dim
    assert results.objects[0].source_b64 is None

    # Test image embedding
    results = kit.embed_image(sample_image)
    assert len(results.objects) == 1
    assert len(results.objects[0].embedding.shape) == expected_dim
    assert isinstance(results.objects[0].source_b64, str)

    # Test single-page PDF
    results = kit.embed_pdf(sample_pdf)
    assert len(results.objects) == 1
    assert len(results.objects[0].embedding.shape) == expected_dim
    assert isinstance(results.objects[0].source_b64, str)

    # Test multi-page PDF
    results = kit.embed_pdf(longer_pdf)
    assert len(results.objects) == 5
    assert len(results.objects[0].embedding.shape) == expected_dim
    assert isinstance(results.objects[0].source_b64, str)


# Setup test files
sample_image = get_sample_image()
sample_pdf = Path("tests/fixtures/2407.01449v6_p1.pdf")
longer_pdf = Path("tests/fixtures/2407.01449v6_p1_p5.pdf")

# Test Cohere provider
print("Testing Cohere provider...")
for input_type in [CohereInputType.SEARCH_QUERY, CohereInputType.SEARCH_DOCUMENT]:
    kit = EmbedKit.cohere(
        model=Model.Cohere.EMBED_V4_0,
        api_key=os.getenv("COHERE_API_KEY"),
        text_batch_size=64,
        image_batch_size=8,
        text_input_type=input_type,
    )
    test_provider(kit, expected_dim=1)

# Test ColPali provider
print("Testing ColPali provider...")
for model in [
    Model.ColPali.COLSMOL_256M,
    Model.ColPali.COLSMOL_500M,
    Model.ColPali.COLPALI_V1_3,
]:
    kit = EmbedKit.colpali(
        model=model,
        text_batch_size=16,
        image_batch_size=8,
    )
    test_provider(kit, expected_dim=2)

# Test Jina provider
print("Testing Jina provider...")
kit = EmbedKit.jina(
    model=Model.Jina.CLIP_V2,
    api_key=os.getenv("JINAAI_API_KEY"),
    text_batch_size=32,
    image_batch_size=8,
)
test_provider(kit, expected_dim=1)
