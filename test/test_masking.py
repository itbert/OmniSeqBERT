import unittest
import torch
import numpy as np
from src.utils.masking import mask_features


class TestMaskingUtils(unittest.TestCase):
    def setUp(self):
        """Подготовка данных"""
        self.batch_size = 2
        self.seq_len = 6
        self.item_cardinality = 100
        self.hidden_dim = 32

        self.features_batch = {
            "item_id": torch.randint(0, self.item_cardinality,
                                     (self.batch_size, self.seq_len)),
            "price": torch.rand(self.batch_size,
                                self.seq_len) * 100.0,
            "category": torch.randint(0,
                                      10,
                                      (self.batch_size,
                                       self.seq_len))
        }
        self.maskable_features = ["item_id", "price"]

    def test_mask_features_basic_functionality(self):
        mask_ratio = 0.3  # ~ 1.8 -> 2
        mask_vals = {"item_id": 0, "price": -1.0}

        masked_batch, masked_positions = mask_features(
            self.features_batch, self.maskable_features, mask_ratio, mask_vals
        )

        for name, orig_tensor in self.features_batch.items():
            self.assertEqual(masked_batch[name].shape, orig_tensor.shape)
        self.assertEqual(masked_positions.shape, (self.batch_size,
                                                  self.seq_len))

        self.assertTrue(torch.equal(masked_batch["category"],
                                    self.features_batch["category"]))

        for name in self.maskable_features:
            orig = self.features_batch[name]
            masked = masked_batch[name]
            mask_val = mask_vals.get(name, 0)

            expected_masked_values = torch.full_like(orig[masked_positions],
                                                     mask_val)
            actual_masked_values = masked[masked_positions]
            self.assertTrue(torch.allclose(actual_masked_values,
                                           expected_masked_values,
                                           equal_nan=bool(np.isnan(mask_val))))

            unmasked_positions = ~masked_positions
            self.assertTrue(torch.equal(masked[unmasked_positions], orig[unmasked_positions]))

        num_expected_masks_per_seq = int(self.seq_len * mask_ratio)
        if num_expected_masks_per_seq == 0 and mask_ratio > 0:
            num_expected_masks_per_seq = 1
        expected_total_masks = self.batch_size * num_expected_masks_per_seq
        self.assertEqual(masked_positions.sum().item(), expected_total_masks)

    def test_mask_features_ratio_zero(self):
        """ratio 0"""
        mask_ratio = 0.0
        mask_vals = {"item_id": 0, "price": -1.0}

        masked_batch, masked_positions = mask_features(
            self.features_batch, self.maskable_features, mask_ratio, mask_vals
        )

        self.assertFalse(masked_positions.any().item())

        for name in self.features_batch:
            self.assertTrue(torch.equal(masked_batch[name],
                                        self.features_batch[name]))

    def test_mask_features_ratio_one(self):
        """ratio 1"""
        mask_ratio = 1.0
        mask_vals = {"item_id": 0, "price": -1.0}

        masked_batch, masked_positions = mask_features(
            self.features_batch, self.maskable_features, mask_ratio, mask_vals
        )

        self.assertTrue(masked_positions.all().item())

        for name in self.maskable_features:
            mask_val = mask_vals.get(name, 0)
            expected_tensor = torch.full_like(self.features_batch[name],
                                              mask_val)
            self.assertTrue(torch.allclose(masked_batch[name],
                                           expected_tensor,
                                           equal_nan=bool(np.isnan(mask_val))))

        self.assertTrue(torch.equal(masked_batch["category"],
                                    self.features_batch["category"]))

    def test_mask_features_small_seq_len_nonzero_ratio(self):
        """seq_len=3 и mask_ratio=0.1"""
        small_seq_len = 3
        mask_ratio = 0.1  # -> 0.3 -> 1
        batch = {
            "item_id": torch.randint(0,
                                     self.item_cardinality,
                                     (self.batch_size, small_seq_len)),
            "price": torch.rand(self.batch_size,
                                small_seq_len) * 100.0,
        }
        maskable = ["item_id", "price"]
        mask_vals = {"item_id": 0, "price": -1.0}

        masked_batch, masked_positions = mask_features(batch, maskable,
                                                       mask_ratio,
                                                       mask_vals)

        expected_num_masks_per_seq = 1  # floor(3 * 0.1) = 0
        self.assertEqual(masked_positions.sum().item(),
                         self.batch_size * expected_num_masks_per_seq)

    def test_mask_features_no_maskable_features(self):
        mask_ratio = 0.2
        mask_vals = {"item_id": 0, "price": -1.0}

        masked_batch, masked_positions = mask_features(
            self.features_batch, [], mask_ratio, mask_vals
        )

        self.assertFalse(masked_positions.any().item())

        for name in self.features_batch:
            self.assertTrue(torch.equal(masked_batch[name],
                                        self.features_batch[name]))

    def test_mask_features_nan_mask_val(self):
        mask_ratio = 0.3
        nan_val = float('nan')
        mask_vals = {"price": nan_val}
        maskable = ["price"]

        masked_batch, masked_positions = mask_features(
            self.features_batch, maskable, mask_ratio, mask_vals
        )

        orig_price = self.features_batch["price"]
        masked_price = masked_batch["price"]

        self.assertTrue(torch.all(torch.isnan(masked_price[masked_positions])))

        unmasked_positions = ~masked_positions
        self.assertTrue(torch.equal(masked_price[unmasked_positions],
                                    orig_price[unmasked_positions]))

        self.assertTrue(torch.equal(masked_batch["item_id"],
                                    self.features_batch["item_id"]))
        self.assertTrue(torch.equal(masked_batch["category"],
                                    self.features_batch["category"]))

    def test_mask_features_default_mask_val(self):
        mask_ratio = 0.3
        maskable = ["item_id"]

        masked_batch, masked_positions = mask_features(
            self.features_batch, maskable, mask_ratio
        )

        orig_item_id = self.features_batch["item_id"]
        masked_item_id = masked_batch["item_id"]

        # На маскированных позициях item_id 0
        expected_masked_values = torch.zeros_like(
            orig_item_id[masked_positions])
        actual_masked_values = masked_item_id[masked_positions]
        self.assertTrue(torch.equal(actual_masked_values,
                                    expected_masked_values))

        # На немаскированных позициях item_id совпадает
        unmasked_positions = ~masked_positions
        self.assertTrue(torch.equal(masked_item_id[unmasked_positions],
                                    orig_item_id[unmasked_positions]))

        # price и category не должны измениться
        self.assertTrue(torch.equal(masked_batch["price"],
                                    self.features_batch["price"]))
        self.assertTrue(torch.equal(masked_batch["category"],
                                    self.features_batch["category"]))


if __name__ == '__main__':
    unittest.main()
