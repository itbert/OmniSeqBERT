import unittest
import tempfile
import os
import torch
from src.omniseqbert.models.omniseqbert import OmniSeqBERT
from src.omniseqbert.models.features_encoders import FeatureMetadata


class TestOmniSeqBERT(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.hidden_dim = 32
        self.max_seq_len = 10

        self.feature_configs = {
            "item_id": FeatureMetadata(name="item_id",
                                       dtype="categorical",
                                       cardinality=100),
            "category": FeatureMetadata(name="category",
                                        dtype="categorical",
                                        cardinality=10),
            "price": FeatureMetadata(name="price",
                                     dtype="numerical",
                                     min_val=0.0,
                                     max_val=100.0),
            "timestamp": FeatureMetadata(name="timestamp",
                                         dtype="datetime"),
        }

        # item_id: LongTensor (B, L)
        self.features_batch = {
            "item_id": torch.randint(0, 100, (self.batch_size, self.seq_len)),
            "category": torch.randint(0, 10, (self.batch_size, self.seq_len)),
            "price": torch.rand(self.batch_size, self.seq_len) * 100.0,
            "timestamp": torch.randint(1600000000,
                                       1700000000,
                                       (self.batch_size,
                                        self.seq_len)).float(),
        }

        self.pad_mask = torch.ones(self.batch_size,
                                   self.seq_len,
                                   dtype=torch.bool)

        # Shape: (B, L)
        self.masked_positions = torch.zeros((self.batch_size,
                                             self.seq_len),
                                            dtype=torch.bool)
        self.masked_positions[0, 1::2] = True  # 1, 3 для первого батча
        self.masked_positions[1, ::2] = True   # 0, 2 для второго батча

    def test_initialization(self):
        model = OmniSeqBERT(
            feature_configs=self.feature_configs,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=["item_id", "category", "price"]
        )
        self.assertIsInstance(model, OmniSeqBERT)
        self.assertEqual(len(model.encoders), len(self.feature_configs))
        self.assertIn('item_id', model.encoders)
        self.assertIn('price', model.encoders)
        self.assertEqual(len(model.transformer_layers), 2)
        self.assertIn('item_id', model.reconstruction_heads)
        self.assertIn('category', model.reconstruction_heads)
        self.assertIn('price', model.reconstruction_heads)
        self.assertNotIn('timestamp', model.reconstruction_heads)

    def test_forward_pass(self):
        model = OmniSeqBERT(
            feature_configs=self.feature_configs,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=list(self.feature_configs.keys())
        )
        model.eval()

        with torch.no_grad():
            output = model(self.features_batch, pad_mask=self.pad_mask)

        expected_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_reconstruct_features(self):
        model = OmniSeqBERT(
            feature_configs=self.feature_configs,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=["item_id", "category", "price"]
        )
        model.eval()

        with torch.no_grad():
            last_hidden_states = model(self.features_batch,
                                       pad_mask=self.pad_mask)
            reconstructed = model.reconstruct_features(last_hidden_states,
                                                       self.masked_positions)

        self.assertIsInstance(reconstructed, dict)
        expected_keys = set(["item_id", "category", "price"])
        self.assertEqual(set(reconstructed.keys()), expected_keys)

        num_items = self.feature_configs["item_id"].cardinality
        expected_item_shape = (self.masked_positions.sum().item(), num_items)
        self.assertEqual(reconstructed["item_id"].shape, expected_item_shape)

        expected_price_shape = (self.masked_positions.sum().item(), 1)
        self.assertEqual(reconstructed["price"].shape, expected_price_shape)

        num_cats = self.feature_configs["category"].cardinality
        expected_cat_shape = (self.masked_positions.sum().item(), num_cats)
        self.assertEqual(reconstructed["category"].shape, expected_cat_shape)

    def test_calculate_loss_categorical(self):
        model = OmniSeqBERT(
            feature_configs=self.feature_configs,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=["item_id"]
        )
        model.train()

        last_hidden_states = model(self.features_batch,
                                   pad_mask=self.pad_mask)
        reconstructed = model.reconstruct_features(last_hidden_states,
                                                   self.masked_positions)

        original_batch = {"item_id": self.features_batch["item_id"]}
        loss = model.calculate_loss(reconstructed,
                                    original_batch,
                                    self.masked_positions)

        self.assertEqual(loss.ndim, 0)  # Скаляр
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_calculate_loss_numerical(self):
        model = OmniSeqBERT(
            feature_configs=self.feature_configs,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=["price"]
        )
        model.train()

        last_hidden_states = model(self.features_batch,
                                   pad_mask=self.pad_mask)
        reconstructed = model.reconstruct_features(last_hidden_states,
                                                   self.masked_positions)

        original_batch = {"price": self.features_batch["price"]}
        loss = model.calculate_loss(reconstructed, original_batch,
                                    self.masked_positions)

        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_save_and_load(self):
        model = OmniSeqBERT(
            feature_configs=self.feature_configs,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=["item_id"]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_model")
            model.save_pretrained(save_path)

            loaded_model = OmniSeqBERT.from_pretrained(
                save_directory=save_path,
                feature_configs=self.feature_configs,
                hidden_dim=self.hidden_dim,
                num_layers=1,
                num_heads=2,
                max_seq_len=self.max_seq_len,
                maskable_features=["item_id"]
            )

            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(p1, p2), "differ after load")

    def test_get_item_embeddings(self):
        model = OmniSeqBERT(
            feature_configs=self.feature_configs,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=["item_id"]
        )

        embeddings = model.get_item_embeddings(item_feature_name="item_id")

        num_items = self.feature_configs["item_id"].cardinality
        expected_shape = (num_items, self.hidden_dim)
        self.assertEqual(embeddings.shape, expected_shape)
        self.assertFalse(torch.allclose(embeddings,
                                        torch.zeros_like(embeddings)))
        non_existent_embeddings = model.get_item_embeddings(
            item_feature_name="non_existent")
        self.assertIsNone(non_existent_embeddings)

        model_no_cat = OmniSeqBERT(
            feature_configs={"price": self.feature_configs["price"]},
            hidden_dim=self.hidden_dim,
            num_layers=1,
            num_heads=2,
            max_seq_len=self.max_seq_len,
            maskable_features=["price"]
        )
        price_embeddings = model_no_cat.get_item_embeddings(
            item_feature_name="price")
        self.assertIsNone(price_embeddings)


if __name__ == '__main__':
    unittest.main()
