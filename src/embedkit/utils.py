from pdf2image import convert_from_path
from pathlib import Path
from .config import get_temp_dir
from typing import Union


def pdf_to_images(pdf_path: Path) -> list[Path]:
    """Convert a PDF file to a list of images."""
    root_temp_dir = get_temp_dir()
    img_temp_dir = root_temp_dir / "images"
    img_temp_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(pdf_path=str(pdf_path), output_folder=str(img_temp_dir))
    image_paths = []

    for i, image in enumerate(images):
        output_path = img_temp_dir / f"{pdf_path.stem}_{i}.png"
        if output_path.exists():
            output_path.unlink()

        image.save(output_path)
        image_paths.append(output_path)
    return image_paths


def image_to_base64(image_path: Union[str, Path]) -> tuple[str, str]:
    """Convert an image to base64 and return the base64 data and content type.

    Args:
        image_path: Path to the image file

    Returns:
        tuple[str, str]: (base64_data, content_type)

    Raises:
        ValueError: If the image cannot be read or has an unsupported format
    """
    import base64

    try:
        base64_data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read image {image_path}: {e}") from e

    if isinstance(image_path, Path):
        image_path_str = str(image_path)

    if image_path_str.lower().endswith(".png"):
        content_type = "image/png"
    elif image_path_str.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif image_path_str.lower().endswith(".gif"):
        content_type = "image/gif"
    else:
        raise ValueError(
            f"Unsupported image format for {image_path}; expected .png, .jpg, .jpeg, or .gif"
        )

    return base64_data, content_type
