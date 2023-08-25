import numpy as np
import torch
from skimage import data

from meddlr.metrics.sdfd import SDFD


def test_SDFD_noise():
    """Test that SDFD increases as Gaussian noise increases."""

    metric = SDFD()
    noise_levels = [0, 0.05]

    target = data.camera().astype(np.float32)
    target = target + 1j * target
    targets = np.repeat(target[np.newaxis, np.newaxis, :, :], len(noise_levels), axis=0)

    preds = np.zeros(targets.shape).astype(np.float32)
    preds = preds + 1j * preds

    for i, noise_level in enumerate(noise_levels):
        pred = (
            target
            + np.random.randn(*target.shape).astype(np.complex64) * noise_level * target.mean()
        )
        preds[i, 0, :, :] = pred
        torch.cuda.empty_cache()

    targets = torch.as_tensor(targets)
    preds = torch.as_tensor(preds)

    metric(preds, targets)
    out = metric.compute().squeeze(-1)
    sorted_out, _ = torch.sort(out, 0)

    assert torch.allclose(sorted_out, out)
