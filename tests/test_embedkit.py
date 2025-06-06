# tests/test_embedkit.py
import os
import pytest
import numpy as np
from pathlib import Path
from embedkit import EmbedKit
from embedkit.models import Model


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
def cohere_kit():
    """Fixture for Cohere kit."""
    return EmbedKit.cohere(
        model=Model.Cohere.EMBED_V4_0,
        api_key=os.getenv("COHERE_API_KEY"),
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
def snowflake_kit():
    """Fixture for Snowflake kit."""
    return EmbedKit.snowflake(
        model=Model.Snowflake.ARCTIC_EMBED_M_V1_5,
    )


# ===============================
# Cohere tests
# ===============================
@pytest.mark.cohere
def test_cohere_document_embedding(cohere_kit):
    """Test document embedding with Cohere models."""
    result = cohere_kit.embed_document("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Cohere"
    assert result.input_type == "search_document"


@pytest.mark.cohere
def test_cohere_query_embedding(cohere_kit):
    """Test query embedding with Cohere models."""
    result = cohere_kit.embed_query("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Cohere"
    assert result.input_type == "search_query"


@pytest.mark.cohere
@pytest.mark.parametrize(
    "embed_method,file_fixture",
    [
        ("embed_image", "sample_image_path"),
        ("embed_pdf", "sample_pdf_path"),
    ],
)
def test_cohere_file_embedding(request, embed_method, file_fixture, cohere_kit):
    """Test file embedding with Cohere model."""
    file_path = request.getfixturevalue(file_fixture)
    embed_func = getattr(cohere_kit, embed_method)
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
        )


@pytest.mark.cohere
def test_cohere_query_vs_document(cohere_kit):
    """Test that query and document embeddings are different for Cohere."""
    text = "Hello world"

    query_result = cohere_kit.embed_query(text)
    doc_result = cohere_kit.embed_document(text)

    # Verify different input types
    assert query_result.input_type == "search_query"
    assert doc_result.input_type == "search_document"

    # Verify different embeddings
    assert not np.array_equal(query_result.objects[0].embedding, doc_result.objects[0].embedding)


# ===============================
# ColPali tests
# ===============================
@pytest.mark.colpali
def test_colpali_document_embedding():
    """Test document embedding with Colpali model."""
    kit = EmbedKit.colpali(model=Model.ColPali.COLPALI_V1_3)
    result = kit.embed_document("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 2
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "ColPali"
    assert result.input_type == "text"


@pytest.mark.colpali
def test_colpali_query_embedding():
    """Test query embedding with Colpali model."""
    kit = EmbedKit.colpali(model=Model.ColPali.COLPALI_V1_3)
    result = kit.embed_query("Hello world")

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


@pytest.mark.colpali
def test_colpali_query_aliases_document():
    """Test that query embeddings alias to document embeddings for ColPali."""
    kit = EmbedKit.colpali(model=Model.ColPali.COLPALI_V1_3)
    text = "Hello world"

    query_result = kit.embed_query(text)
    doc_result = kit.embed_document(text)

    # Verify same input type
    assert query_result.input_type == "text"
    assert doc_result.input_type == "text"

    # Verify same embeddings
    assert np.array_equal(query_result.objects[0].embedding, doc_result.objects[0].embedding)


# ===============================
# Jina tests
# ===============================
@pytest.mark.jina
def test_jina_document_embedding(jina_kit):
    """Test document embedding with Jina model."""
    result = jina_kit.embed_document("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Jina"
    assert result.input_type == "text"


@pytest.mark.jina
def test_jina_query_embedding(jina_kit):
    """Test query embedding with Jina model."""
    result = jina_kit.embed_query("Hello world")

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


@pytest.mark.jina
def test_jina_query_aliases_document(jina_kit):
    """Test that query embeddings alias to document embeddings for Jina."""
    text = "Hello world"

    query_result = jina_kit.embed_query(text)
    doc_result = jina_kit.embed_document(text)

    # Verify same input type
    assert query_result.input_type == "text"
    assert doc_result.input_type == "text"

    # Verify same embeddings
    assert np.array_equal(query_result.objects[0].embedding, doc_result.objects[0].embedding)


# ===============================
# Snowflake tests
# ===============================
@pytest.mark.snowflake
@pytest.mark.parametrize(
    "model",
    [
        Model.Snowflake.ARCTIC_EMBED_M_V1_5,
        Model.Snowflake.ARCTIC_EMBED_L_V2_0,
    ],
)
def test_snowflake_document_embedding(model):
    """Test document embedding with Snowflake models."""
    kit = EmbedKit.snowflake(model=model)
    result = kit.embed_document("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Snowflake"
    assert result.input_type is None


@pytest.mark.snowflake
@pytest.mark.parametrize(
    "model",
    [
        Model.Snowflake.ARCTIC_EMBED_M_V1_5,
        Model.Snowflake.ARCTIC_EMBED_L_V2_0,
    ],
)
def test_snowflake_query_embedding(model):
    """Test query embedding with Snowflake models."""
    kit = EmbedKit.snowflake(model=model)
    result = kit.embed_query("Hello world")

    assert len(result.objects) == 1
    assert len(result.objects[0].embedding.shape) == 1
    assert result.objects[0].source_b64 is None
    assert result.model_provider == "Snowflake"
    assert result.input_type == "query"


@pytest.mark.snowflake
def test_snowflake_image_embedding(snowflake_kit):
    """Test that image embedding raises appropriate error."""
    with pytest.raises(Exception) as exc_info:
        snowflake_kit.embed_image("dummy_path")
    assert "does not support image embeddings" in str(exc_info.value)


@pytest.mark.snowflake
def test_snowflake_pdf_embedding(snowflake_kit):
    """Test that PDF embedding raises appropriate error."""
    with pytest.raises(Exception) as exc_info:
        snowflake_kit.embed_pdf("dummy_path")
    assert "does not support image embeddings" in str(exc_info.value)


@pytest.mark.snowflake
def test_snowflake_invalid_model():
    """Test that invalid model raises appropriate error."""
    with pytest.raises(ValueError):
        EmbedKit.snowflake(model="invalid_model")


@pytest.mark.snowflake
def test_snowflake_query_vs_document(snowflake_kit):
    """Test that query and document embeddings are different for Snowflake."""
    text = "Hello world"

    query_result = snowflake_kit.embed_query(text)
    doc_result = snowflake_kit.embed_document(text)

    # Verify different input types
    assert query_result.input_type == "query"
    assert doc_result.input_type is None

    # Verify different embeddings
    assert not np.array_equal(query_result.objects[0].embedding, doc_result.objects[0].embedding)
