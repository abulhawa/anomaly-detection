"""PyTorch datasets used by the modelling notebooks."""

from collections.abc import Callable

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class MVTecImageDataset(Dataset[tuple[Tensor, int, str]]):
    """Load MVTec images described by a manifest dataframe for PyTorch.

    Parameters
    ----------
    frame:
        A subset of the dataframe returned by :func:`build_mvtec_manifest`.
        It must contain a path to each image (``path``) and the binary anomaly
        label (``is_anomaly``). The class keeps only these two columns, so its
        output contract is identical for training, validation, and test data.
    transform:
        A callable that receives one RGB :class:`PIL.Image.Image` and returns
        the tensor consumed by a model, normally a ``float32`` tensor with
        shape ``(channels, height, width)``. The dataset deliberately does not
        choose a resize, normalization, or augmentation itself: those choices
        belong to the experiment and model, not to the dataset loader.

        For example, the frozen ResNet-18 baseline uses a 224 x 224 resize,
        conversion to a tensor, and ImageNet normalization because its frozen
        weights were pretrained with that convention. The autoencoder uses a
        256 x 256 resize and ``ToTensor()`` only because it learns directly
        from pixels in the [0, 1] range. Keeping these transforms outside the
        class lets both notebooks share the same loading behaviour without
        silently applying the wrong preprocessing.

    Returns
    -------
    tuple[Tensor, int, str]
        ``(image_tensor, anomaly_label, path)``. ``anomaly_label`` is ``0``
        for a good image and ``1`` for an anomalous image. The path is kept for
        visualisation and later error analysis.

    Notes
    -----
    Images are converted to RGB before the transform is applied. This gives
    all models three channels, including when a MVTec category is stored as a
    grayscale image. A transform used for validation or test data should be
    deterministic. Any random augmentation should be documented and normally
    restricted to the normal training dataset.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        required_columns = {"path", "is_anomaly"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                "frame is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        self.frame = frame.loc[:, ["path", "is_anomaly"]].reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        row = self.frame.iloc[index]
        path = str(row["path"])
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))

        return image_tensor, int(row["is_anomaly"]), path
