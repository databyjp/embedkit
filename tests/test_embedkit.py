import os
import pytest
from pathlib import Path
from embedkit import EmbedKit
from embedkit.models import Model
from embedkit.providers.cohere import CohereInputType


# Fixture for sample image
@pytest.fixture
def sample_image():
    """Fixture to provide a sample image for testing."""
    url = "https://upload.wikimedia.org/wikipedia/commons/b/b8/English_Wikipedia_HomePage_2001-12-20.png"
    import requests
    from tempfile import NamedTemporaryFile

    headers = {"User-Agent": "EmbedKit-Example/1.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    temp_file = NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(response.content)
    temp_file.close()

    return Path(temp_file.name)


# Colpali tests
def test_colpali_text_embedding():
    """Test text embedding with Colpali model."""
    kit = EmbedKit.colpali(model=Model.COLPALI_V1_3)
    embeddings = kit.embed_text("Hello world")

    assert embeddings.shape[0] == 1
    assert len(embeddings.shape) == 3


def test_colpali_image_embedding(sample_image):
    """Test image embedding with Colpali model."""
    kit = EmbedKit.colpali(model=Model.COLPALI_V1_3)
    embeddings = kit.embed_image(sample_image)

    assert embeddings.shape[0] == 1
    assert len(embeddings.shape) == 3


# Cohere tests
@pytest.fixture
def cohere_kit_search_query():
    """Fixture for Cohere kit with search query input type."""
    return EmbedKit.cohere(
        model=Model.COHERE_V4_0,
        api_key=os.getenv("COHERE_API_KEY"),
        text_input_type=CohereInputType.SEARCH_QUERY,
    )


@pytest.fixture
def cohere_kit_search_document():
    """Fixture for Cohere kit with search document input type."""
    return EmbedKit.cohere(
        model=Model.COHERE_V4_0,
        api_key=os.getenv("COHERE_API_KEY"),
        text_input_type=CohereInputType.SEARCH_DOCUMENT,
    )


def test_cohere_search_query_text_embedding(cohere_kit_search_query):
    """Test text embedding with Cohere search query model."""
    embeddings = cohere_kit_search_query.embed_text("Hello world")

    assert embeddings.shape[0] == 1
    assert len(embeddings.shape) == 2


def test_cohere_search_document_text_embedding(cohere_kit_search_document):
    """Test text embedding with Cohere search document model."""
    embeddings = cohere_kit_search_document.embed_text("Hello world")

    assert embeddings.shape[0] == 1
    assert len(embeddings.shape) == 2


def test_cohere_search_document_image_embedding(
    cohere_kit_search_document, sample_image
):
    """Test image embedding with Cohere search document model."""
    embeddings = cohere_kit_search_document.embed_image(sample_image)

    assert embeddings.shape[0] == 1
    assert len(embeddings.shape) == 2


# Error cases
def test_colpali_invalid_model():
    """Test that invalid model raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.colpali(model="invalid_model")


def test_cohere_missing_api_key():
    """Test that missing API key raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.cohere(
            model=Model.COHERE_V4_0,
            api_key=None,
            text_input_type=CohereInputType.SEARCH_QUERY,
        )
