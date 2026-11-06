import gc
import json
import unittest

import torch
from torch import nn

from ..model.generator import UNetSmall
from ..model.training_models import (
    FCNGenerator, GlobalQSMDiscriminator, MODEL_NAMES, Pix2Pix70Discriminator,
    Pix2PixQSMGenerator, create_model,
)


class TrainingModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        gc.collect()

    def test_selection_metadata_and_legacy_generator(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                generator, discriminator, metadata = create_model(name, image_size=32)
                json.dumps(metadata, allow_nan=False)
                self.assertEqual(metadata["model_name"], name)
                self.assertGreater(metadata["parameter_counts"]["generator"]["active_trainable"], 0)
                self.assertFalse(any(isinstance(layer, (nn.Sigmoid, nn.Tanh))
                                     for layer in generator.modules()))
                if name in ("swixqsm_gan", "plain_unet"):
                    self.assertIs(type(generator), UNetSmall)
                    inactive = metadata["parameter_counts"]["generator"]["known_inactive_registered"]
                    self.assertEqual(inactive, 172256)
                    self.assertIn("up1.deconv.weight", generator.state_dict())
                if name == "plain_unet":
                    self.assertIsNone(discriminator)
                    self.assertIsNone(metadata["discriminator"])
                    self.assertEqual(metadata["parameter_counts"]["discriminator"]["registered"], 0)
                del generator, discriminator
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            create_model("unet_typo")

    def test_all_four_real_forward_backward_finite(self):
        # Full published/default widths; no smaller test-only model substitution.
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                torch.manual_seed(14)
                generator, discriminator, _ = create_model(name, image_size=32)
                image = torch.randn(1, 1, 32, 32)
                target = torch.randn_like(image)
                prediction = generator(image)
                self.assertEqual(prediction.shape, target.shape)
                loss = (prediction - target).abs().mean()
                if discriminator is not None:
                    logits = discriminator(image, prediction)
                    self.assertEqual(logits.shape[:2], (1, 1))
                    loss = loss + 0.01 * logits.square().mean()
                loss.backward()
                active_gradients = [parameter.grad for parameter in generator.parameters()
                                    if parameter.grad is not None]
                self.assertGreater(len(active_gradients), 0)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in active_gradients))
                self.assertGreater(sum(float(gradient.abs().sum()) for gradient in active_gradients), 0.)
                if discriminator is not None:
                    disc_gradients = [parameter.grad for parameter in discriminator.parameters()]
                    self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                        for gradient in disc_gradients))
                if isinstance(generator, UNetSmall):
                    self.assertTrue(all(parameter.grad is None for name_, parameter in generator.named_parameters()
                                        if ".deconv." in name_))
                del generator, discriminator, prediction, loss, active_gradients

    def test_linear_heads_preserve_negative_and_above_one(self):
        for name in MODEL_NAMES:
            generator, discriminator, _ = create_model(name, image_size=32)
            generator.eval()
            if isinstance(generator, UNetSmall):
                head = generator.final[0]
            elif isinstance(generator, FCNGenerator):
                head = generator.final
            else:
                head = generator.decoders[-1][1]
            with torch.no_grad():
                head.weight.zero_()
                for bias in (-2.5, 3.5):
                    head.bias.fill_(bias)
                    result = generator(torch.zeros(1, 1, 32, 40))
                    torch.testing.assert_close(result, torch.full_like(result, bias), rtol=0., atol=0.)
            del generator, discriminator

    def test_global_discriminator_is_qsm_only_and_unrestricted(self):
        discriminator = GlobalQSMDiscriminator(image_size=32).eval()
        qsm = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            first = discriminator(torch.zeros_like(qsm), qsm)
            second = discriminator(torch.randn_like(qsm) * 100, qsm)
            torch.testing.assert_close(first, second, rtol=0., atol=0.)
            self.assertEqual(first.shape, (2, 1))
            discriminator.classifier.weight.zero_()
            discriminator.classifier.bias.fill_(-3.)
            self.assertTrue((discriminator(qsm, qsm) == -3).all())
        self.assertFalse(any(isinstance(layer, (nn.BatchNorm2d, nn.Sigmoid, nn.ReLU,
                                                nn.AdaptiveAvgPool2d))
                             for layer in discriminator.modules()))
        norms = [layer for layer in discriminator.modules() if isinstance(layer, nn.InstanceNorm2d)]
        self.assertEqual(len(norms), 3)
        self.assertTrue(all(not layer.affine and not layer.track_running_stats for layer in norms))
        convolutions = [layer for layer in discriminator.features if isinstance(layer, nn.Conv2d)]
        self.assertEqual([layer.kernel_size[0] for layer in convolutions], [3] * 4)
        self.assertEqual([layer.stride[0] for layer in convolutions], [2, 2, 2, 1])
        self.assertEqual(discriminator.classifier.in_features, 16)

    def test_fcn_decoder_has_no_skip_input_or_transposed_convolution(self):
        generator = FCNGenerator().eval()
        first_decoder_conv = next(layer for layer in generator.decoder.modules() if isinstance(layer, nn.Conv2d))
        self.assertEqual(first_decoder_conv.in_channels, 256)
        self.assertFalse(any(isinstance(layer, nn.ConvTranspose2d) for layer in generator.modules()))
        decoder_convolutions = [layer for layer in generator.decoder.modules() if isinstance(layer, nn.Conv2d)]
        # Every decoder stage consumes only the previous stage's channels.
        self.assertEqual([(layer.in_channels, layer.out_channels) for layer in decoder_convolutions],
                         [(256, 128), (128, 128), (128, 64), (64, 64), (64, 32), (32, 32)])

    def test_pix2pix_default_416_grid_padding_and_dropouts(self):
        generator, _, metadata = create_model("pix2pix", image_size=416)
        generator.eval()
        self.assertEqual(metadata["image_size"], 416)
        self.assertEqual(sum(isinstance(layer, nn.Dropout) for layer in generator.modules()), 3)
        seen = []
        handle = generator.encoders[0].register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].shape))
        with torch.no_grad():
            result = generator(torch.randn(1, 1, 416, 416))
        handle.remove()
        self.assertEqual(seen, [torch.Size((1, 1, 512, 512))])
        self.assertEqual(result.shape, (1, 1, 416, 416))
        self.assertTrue(torch.isfinite(result).all())

    def test_pix2pix_discriminator_nominal_rf70_and_conditioning(self):
        discriminator = Pix2Pix70Discriminator().eval()
        receptive_field, jump = 1, 1
        convolutions = [layer for layer in discriminator.net if isinstance(layer, nn.Conv2d)]
        self.assertEqual([layer.stride[0] for layer in convolutions], [2, 2, 2, 1, 1])
        self.assertEqual([layer.kernel_size[0] for layer in convolutions], [4] * 5)
        for layer in convolutions:
            receptive_field += (layer.kernel_size[0] - 1) * jump
            jump *= layer.stride[0]
        self.assertEqual(receptive_field, 70)
        self.assertEqual(jump, 8)
        swi, qsm = torch.zeros(1, 1, 32, 40), torch.ones(1, 1, 32, 40)
        seen = []
        handle = convolutions[0].register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].clone()))
        with torch.no_grad():
            result = discriminator(swi, qsm)
        handle.remove()
        torch.testing.assert_close(seen[0], torch.cat((swi, qsm), dim=1))
        self.assertEqual(result.shape, (1, 1, 2, 3))

    def test_common_grid_contract_rejects_unsupported_input(self):
        for generator in (FCNGenerator(), Pix2PixQSMGenerator()):
            with self.assertRaisesRegex(ValueError, "multiples of 8"):
                generator(torch.zeros(1, 1, 33, 40))
            with self.assertRaisesRegex(ValueError, "H,W >= 32"):
                generator(torch.zeros(1, 1, 16, 16))
            del generator
        discriminator = GlobalQSMDiscriminator(image_size=32)
        with self.assertRaisesRegex(ValueError, "identical shape"):
            discriminator(torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 40))
        with self.assertRaisesRegex(ValueError, "configured square"):
            discriminator(torch.zeros(1, 1, 32, 40), torch.zeros(1, 1, 32, 40))
        for size in (31, 33, True, 32.0):
            with self.assertRaisesRegex(ValueError, "image_size"):
                create_model("swixqsm_gan", image_size=size)

    def test_global_discriminator_preserves_spatial_classifier_for_416(self):
        _, discriminator, metadata = create_model("swixqsm_gan", image_size=416)
        self.assertEqual(discriminator.classifier.in_features, 52 * 52)
        self.assertEqual(metadata["discriminator"]["flattened_spatial_features"], 52 * 52)
        discriminator.eval()
        with torch.no_grad():
            image = torch.randn(1, 1, 416, 416)
            result = discriminator(image, image)
        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
