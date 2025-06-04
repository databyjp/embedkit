# ./main.py
from embedkit import EmbedKit
from embedkit.classes import Model, CohereInputType
from pathlib import Path
import os
import numpy as np


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


def print_embedding_stats(embedding: np.ndarray, prefix: str = ""):
    """Print statistics about an embedding."""
    print(f"{prefix}Embedding shape: {embedding.shape}")
    print(f"{prefix}First 5 values: {embedding[:5]}")
    print(f"{prefix}Last 5 values: {embedding[-5:]}")
    print()


def test_provider(kit: EmbedKit, expected_dim: int, supports_images: bool = True):
    """Test a provider with various inputs."""
    print(f"\nTesting {kit.provider_info}...")

    # Test text embedding
    test_texts = [
        "Hello world",
        "This is a longer text that should generate a different embedding",
        "The quick brown fox jumps over the lazy dog",
    ]
    results = kit.embed_text(test_texts)
    assert len(results.objects) == len(test_texts)

    print("Text Embedding Results:")
    for i, (text, obj) in enumerate(zip(test_texts, results.objects)):
        print(f"\nText {i+1}: {text}")
        print_embedding_stats(obj.embedding, prefix="  ")
        assert len(obj.embedding.shape) == expected_dim
        assert obj.source_b64 is None

    if supports_images:
        # Test image embedding
        results = kit.embed_image(sample_image)
        assert len(results.objects) == 1
        print("\nImage Embedding Results:")
        print_embedding_stats(results.objects[0].embedding, prefix="  ")
        assert len(results.objects[0].embedding.shape) == expected_dim
        assert isinstance(results.objects[0].source_b64, str)

        # Test single-page PDF
        results = kit.embed_pdf(sample_pdf)
        assert len(results.objects) == 1
        print("\nSingle-page PDF Embedding Results:")
        print_embedding_stats(results.objects[0].embedding, prefix="  ")
        assert len(results.objects[0].embedding.shape) == expected_dim
        assert isinstance(results.objects[0].source_b64, str)

        # Test multi-page PDF
        results = kit.embed_pdf(longer_pdf)
        assert len(results.objects) == 5
        print("\nMulti-page PDF Embedding Results (first page):")
        print_embedding_stats(results.objects[0].embedding, prefix="  ")
        assert len(results.objects[0].embedding.shape) == expected_dim
        assert isinstance(results.objects[0].source_b64, str)
    else:
        # Test that image embedding raises error
        try:
            kit.embed_image(sample_image)
            assert False, "Image embedding should raise error"
        except Exception as e:
            assert "does not support image embeddings" in str(e)
            print("\nSuccessfully caught image embedding error:", str(e))

        # Test that PDF embedding raises error
        try:
            kit.embed_pdf(sample_pdf)
            assert False, "PDF embedding should raise error"
        except Exception as e:
            assert "does not support image embeddings" in str(e)
            print("\nSuccessfully caught PDF embedding error:", str(e))


# Setup test files
sample_image = get_sample_image()
sample_pdf = Path("tests/fixtures/2407.01449v6_p1.pdf")
longer_pdf = Path("tests/fixtures/2407.01449v6_p1_p5.pdf")

# Test Cohere provider
print("\n=== Testing Cohere provider ===")
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
print("\n=== Testing ColPali provider ===")
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
print("\n=== Testing Jina provider ===")
kit = EmbedKit.jina(
    model=Model.Jina.CLIP_V2,
    api_key=os.getenv("JINAAI_API_KEY"),
    text_batch_size=32,
    image_batch_size=8,
)
test_provider(kit, expected_dim=1)

# Test Snowflake provider
print("\n=== Testing Snowflake provider ===")
kit = EmbedKit.snowflake(
    model=Model.Snowflake.ARCTIC_EMBED_M_V1_5,
    text_batch_size=32,
    device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else None,
)
test_provider(kit, expected_dim=1, supports_images=False)
