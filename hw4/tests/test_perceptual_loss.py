import unittest
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_build_loss():
    from src.hw4.engine import build_loss

    return build_loss


class PerceptualLossTest(unittest.TestCase):
    def test_build_loss_supports_mse_perceptual(self) -> None:
        build_loss = load_build_loss()
        criterion = build_loss(
            "mse_perceptual",
            edge_weight=0.0,
            perceptual_weight=0.1,
            perceptual_pretrained=False,
        )

        prediction = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)
        loss = criterion(prediction, target)

        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_reference_preset_supports_vgg16(self) -> None:
        build_loss = load_build_loss()
        criterion = build_loss(
            "l1_psnr_ssim_perceptual",
            edge_weight=0.0,
            perceptual_weight=0.01,
            perceptual_pretrained=False,
            perceptual_backbone="vgg16",
            perceptual_layer_preset="reference",
        )

        self.assertEqual(criterion.selected_layers, {3, 8, 15})

    def test_perceptual_features_are_frozen(self) -> None:
        build_loss = load_build_loss()
        criterion = build_loss(
            "mse_perceptual",
            edge_weight=0.0,
            perceptual_weight=0.1,
            perceptual_pretrained=False,
        )

        trainable = [
            parameter.requires_grad
            for parameter in criterion.perceptual_extractor.parameters()
        ]

        self.assertTrue(trainable)
        self.assertTrue(all(flag is False for flag in trainable))


if __name__ == "__main__":
    unittest.main()
