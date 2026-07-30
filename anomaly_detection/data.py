"""Dataset discovery helpers for the MVTec AD dataset."""

from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png"})
MANIFEST_COLUMNS = [
    "path",
    "image_id",
    "product",
    "split",
    "defect_type",
    "is_anomaly",
    "width",
    "height",
    "mode",
    "mask_path"
]


def _read_image_metadata(path: Path) -> tuple[int, int, str]:
    """Read an image's dimensions and color mode without decoding all pixels."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read image metadata for {path}") from error

    return width, height, mode


def build_mvtec_manifest(root: str | Path) -> pd.DataFrame:
    """Build an image-only manifest for an extracted MVTec AD dataset.

    The returned table contains the train and test input images. Ground-truth
    masks are not included as samples; defective images instead receive an
    optional ``mask_path`` so masks remain available for later localization or
    error analysis.

    Parameters
    ----------
    root
        Directory containing the MVTec product directories, such as
        ``bottle/``, ``cable/``, and ``capsule/``.

    Returns
    -------
    pandas.DataFrame
        One row per train or test image, in deterministic path order.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    NotADirectoryError
        If ``root`` is not a directory.
    ValueError
        If an image cannot be inspected or no images are discovered.
    """
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"MVTec dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"MVTec dataset path is not a directory: {root}")
    root = root.resolve()

    rows: list[dict[str, object]] = []

    for product_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        product = product_dir.name

        for split in ("train", "test"):
            split_dir = product_dir / split
            if not split_dir.is_dir():
                continue

            defect_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
            for defect_dir in defect_dirs:
                defect_type = defect_dir.name
                is_anomaly = defect_type != "good"

                image_paths = sorted(
                    path
                    for path in defect_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                for image_path in image_paths:
                    width, height, mode = _read_image_metadata(image_path)
                    image_id = image_path.stem
                    expected_mask_path = (
                        product_dir
                        / "ground_truth"
                        / defect_type
                        / f"{image_id}_mask.png"
                    )

                    rows.append(
                        {
                            "path": str(image_path),
                            "image_id": image_id,
                            "product": product,
                            "split": split,
                            "defect_type": defect_type,
                            "is_anomaly": is_anomaly,
                            "width": width,
                            "height": height,
                            "mode": mode,
                            "mask_path": (
                                str(expected_mask_path)
                                if expected_mask_path.is_file()
                                else None
                            ),
                        }
                    )

    if not rows:
        raise ValueError(f"No MVTec images were found below: {root}")

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
