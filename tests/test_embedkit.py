# tests/test_embedkit.py
import os
import pytest
import numpy as np
from pathlib import Path
from embedkit import EmbedKit
from embedkit.models import Model
from embedkit.providers.cohere import CohereInputType
from embedkit.providers.snowflake import SnowflakeInputType


# Fixture for sample image
@pytest.fixture
def sample_image_path():
    """Fixture to provide a sample image for testing."""
    path = Path("tests/fixtures/2407.01449v6_p1.png")
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


# Fixture for sample PDF
@pytest.fixture
def sample_pdf_path():
    """Fixture to provide a sample PDF for testing."""
    path = Path("tests/fixtures/2407.01449v6_p1.pdf")
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path


# Cohere fixtures
@pytest.fixture
def cohere_kit_search_query():
    """Fixture for Cohere kit with search query input type."""
    return EmbedKit.cohere(
        model=Model.Cohere.EMBED_V4_0,
        api_key=os.getenv("COHERE_API_KEY"),
        text_input_type=CohereInputType.SEARCH_QUERY,
    )


@pytest.fixture
def cohere_kit_search_document():
    """Fixture for Cohere kit with search document input type."""
    return EmbedKit.cohere(
        model=Model.Cohere.EMBED_V4_0,
        api_key=os.getenv("COHERE_API_KEY"),
        text_input_type=CohereInputType.SEARCH_DOCUMENT,
    )


# Jina fixtures
@pytest.fixture
def jina_kit():
    """Fixture for Jina kit."""
    return EmbedKit.jina(
        model=Model.Jina.CLIP_V2,
        api_key=os.getenv("JINAAI_API_KEY"),
    )


# Snowflake fixtures
@pytest.fixture
def snowflake_kit_query():
    """Fixture for Snowflake kit with query input type."""
    return EmbedKit.snowflake(
        model=Model.Snowflake.ARCTIC_EMBED_M_V1_5,
        text_input_type=SnowflakeInputType.QUERY,
    )


@pytest.fixture
def snowflake_kit_document():
    """Fixture for Snowflake kit with document input type."""
    return EmbedKit.snowflake(
        model=Model.Snowflake.ARCTIC_EMBED_M_V1_5,
        text_input_type=SnowflakeInputType.DOCUMENT,
    )


# ===============================
# Cohere tests
# ===============================
@pytest.mark.cohere
@pytest.mark.parametrize(
    "cohere_kit_fixture", ["cohere_kit_search_query", "cohere_kit_search_document"]
)
def test_cohere_text_embedding(request, cohere_kit_fixture):
    """Test text embedding with Cohere models."""
    kit = request.getfixturevalue(cohere_kit_fixture)
    result = kit.embed_text("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Cohere"
    assert result.input_type in ["search_query", "search_document"]


@pytest.mark.cohere
@pytest.mark.parametrize(
    "embed_method,file_fixture",
    [
        ("embed_image", "sample_image_path"),
        ("embed_pdf", "sample_pdf_path"),
    ],
)
def test_cohere_search_document_file_embedding(
    request, embed_method, file_fixture, cohere_kit_search_document
):
    """Test file embedding with Cohere search document model."""
    file_path = request.getfixturevalue(file_fixture)
    embed_func = getattr(cohere_kit_search_document, embed_method)
    result = embed_func(file_path)

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.model_provider == "Cohere"
    assert result.input_type == "image"
    if hasattr(result.objects[0], "source_b64"):
        assert result.objects[0].source_b64 is not None


@pytest.mark.cohere
def test_cohere_invalid_model():
    """Test that invalid model raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.cohere(
            model="invalid_model",
            api_key=os.getenv("COHERE_API_KEY"),
        )


@pytest.mark.cohere
def test_cohere_missing_api_key():
    """Test that missing API key raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.cohere(
            model=Model.Cohere.EMBED_V4_0,
            api_key=None,
            text_input_type=CohereInputType.SEARCH_QUERY,
        )


# ===============================
# ColPali tests
# ===============================
@pytest.mark.colpali
def test_colpali_text_embedding():
    """Test text embedding with Colpali model."""
    kit = EmbedKit.colpali(model=Model.ColPali.COLPALI_V1_3)
    result = kit.embed_text("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 2
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "ColPali"
    assert result.input_type == "text"


@pytest.mark.colpali
@pytest.mark.parametrize(
    "embed_method,file_fixture",
    [
        ("embed_image", "sample_image_path"),
        ("embed_pdf", "sample_pdf_path"),
    ],
)
def test_colpali_file_embedding(request, embed_method, file_fixture):
    """Test file embedding with Colpali model."""
    kit = EmbedKit.colpali(model=Model.ColPali.COLPALI_V1_3)
    file_path = request.getfixturevalue(file_fixture)
    embed_func = getattr(kit, embed_method)
    result = embed_func(file_path)

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 2
    assert isinstance(result.objects[0].source_b64, str)
    assert result.model_provider == "ColPali"
    assert result.input_type == "image"


@pytest.mark.colpali
def test_colpali_invalid_model():
    """Test that invalid model raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.colpali(model="invalid_model")


# ===============================
# Jina tests
# ===============================
@pytest.mark.jina
def test_jina_text_embedding(jina_kit):
    """Test text embedding with Jina model."""
    result = jina_kit.embed_text("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Jina"
    assert result.input_type == "text"


@pytest.mark.jina
@pytest.mark.parametrize(
    "embed_method,file_fixture",
    [
        ("embed_image", "sample_image_path"),
        ("embed_pdf", "sample_pdf_path"),
    ],
)
def test_jina_file_embedding(request, embed_method, file_fixture, jina_kit):
    """Test file embedding with Jina model."""
    file_path = request.getfixturevalue(file_fixture)
    embed_func = getattr(jina_kit, embed_method)
    result = embed_func(file_path)

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert isinstance(result.objects[0].source_b64, str)
    assert result.model_provider == "Jina"
    assert result.input_type == "image"


@pytest.mark.jina
def test_jina_invalid_model():
    """Test that invalid model raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.jina(
            model="invalid_model",
            api_key=os.getenv("JINAAI_API_KEY"),
        )


@pytest.mark.jina
def test_jina_missing_api_key():
    """Test that missing API key raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.jina(
            model=Model.Jina.CLIP_V2,
            api_key=None,
        )


# ===============================
# Snowflake tests
# ===============================
@pytest.mark.snowflake
@pytest.mark.parametrize(
    "model,input_type",
    [
        (Model.Snowflake.ARCTIC_EMBED_M_V1_5, SnowflakeInputType.QUERY),
        (Model.Snowflake.ARCTIC_EMBED_M_V1_5, SnowflakeInputType.DOCUMENT),
        (Model.Snowflake.ARCTIC_EMBED_L_V2_0, SnowflakeInputType.QUERY),
        (Model.Snowflake.ARCTIC_EMBED_L_V2_0, SnowflakeInputType.DOCUMENT),
    ],
)
def test_snowflake_text_embedding(request, model, input_type):
    """Test text embedding with Snowflake models."""
    kit = EmbedKit.snowflake(
        model=model,
        text_input_type=input_type,
    )
    result = kit.embed_text("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Snowflake"
    assert result.input_type in ["query", None]


@pytest.mark.snowflake
def test_snowflake_image_embedding(snowflake_kit_query):
    """Test that image embedding raises appropriate error."""
    with pytest.raises(Exception) as exc_info:
        snowflake_kit_query.embed_image("dummy_path")
    assert "does not support image embeddings" in str(exc_info.value)


@pytest.mark.snowflake
def test_snowflake_pdf_embedding(snowflake_kit_query):
    """Test that PDF embedding raises appropriate error."""
    with pytest.raises(Exception) as exc_info:
        snowflake_kit_query.embed_pdf("dummy_path")
    assert "does not support image embeddings" in str(exc_info.value)


@pytest.mark.snowflake
def test_snowflake_invalid_model():
    """Test that invalid model raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.snowflake(
            model="invalid_model",
            text_input_type=SnowflakeInputType.QUERY,
        )
