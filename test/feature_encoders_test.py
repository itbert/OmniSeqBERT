# tests/test_models/test_feature_encoders.py (обновленные тесты)

import pytest
import torch
from unittest.mock import patch, MagicMock
from src.omniseqbert.models.features_encoders import (
    FeatureMetadata,
    BaseFeatureEncoder,
    NumericalFeatureEncoder,
    CategoricalFeatureEncoder,
    DateTimeFeatureEncoder,
    TextFeatureEncoder,
    PretrainedEmbeddingEncoder,
    ENCODER_REGISTRY,
    get_encoder_class,
)


class TestFeatureMetadata:
    def test_feature_metadata_creation(self):
        meta = FeatureMetadata(
            name="test_feature",
            dtype="numerical",
            cardinality=None,
            min_val=0.0,
            max_val=10.0
        )
        assert meta.name == "test_feature"
        assert meta.dtype == "numerical"
        assert meta.cardinality is None
        assert meta.min_val == 0.0
        assert meta.max_val == 10.0


class TestBaseFeatureEncoder:
    """Тесты для BaseFeatureEncoder"""

    def test_abstract_methods(self):
        with pytest.raises(TypeError,
                           match="Can't instantiate abstract class"):
            BaseFeatureEncoder(embedding_dim=10, 
                               feature_metadata=FeatureMetadata(
                                   name="test",
                                   dtype="numerical"))


class TestNumericalFeatureEncoder:
    """Тесты для NumericalFeatureEncoder"""
    @pytest.fixture
    def feature_meta(self):
        return FeatureMetadata(
            name="num_feature",
            dtype="numerical",
            min_val=0.0,
            max_val=10.0
        )

    def test_min_max_normalization(self, feature_meta):
        encoder = NumericalFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            normalization='min_max'
        )
        x = torch.tensor([0.0, 5.0, 10.0])
        embedded = encoder(x)
        assert embedded.shape == (3, 16)

    def test_standard_normalization(self, feature_meta):
        encoder = NumericalFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            normalization='standard'
        )
        x = torch.tensor([0.0, 5.0, 10.0])
        embedded = encoder(x)
        assert embedded.shape == (3, 16)

    def test_adaptive_normalization(self, feature_meta):
        encoder = NumericalFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            normalization='adaptive'
        )
        x = torch.tensor([0.0, 5.0, 10.0])
        embedded = encoder(x)
        assert embedded.shape == (3, 16)

    def test_missing_value_special_token(self, feature_meta):
        encoder = NumericalFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            missing_value_strategy='special_token'
        )
        x_with_nan = torch.tensor([1.0, float('nan'), 3.0])
        embedded = encoder(x_with_nan)
        assert embedded.shape == (3, 16)
        assert not torch.allclose(embedded[0], embedded[1])
        assert torch.allclose(embedded[1], encoder.missing_token_embedding, atol=1e-6)

    def test_get_output_dim(self, feature_meta):
        encoder = NumericalFeatureEncoder(
            embedding_dim=32,
            feature_metadata=feature_meta
        )
        assert encoder.get_output_dim() == 32


class TestCategoricalFeatureEncoder:
    """Тесты для CategoricalFeatureEncoder"""

    @pytest.fixture
    def feature_meta(self):
        return FeatureMetadata(
            name="cat_feature",
            dtype="categorical",
            cardinality=100
        )

    def test_basic_encoding(self, feature_meta):
        encoder = CategoricalFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta
        )
        x = torch.tensor([0, 50, 99])
        embedded = encoder(x)
        assert embedded.shape == (3, 16)

    def test_oov_with_token_strategy(self, feature_meta):
        encoder = CategoricalFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            oov_strategy='token'
        )
        x_with_oov = torch.tensor([0, 150])  # 150 >= cardinality
        embedded = encoder(x_with_oov)
        assert embedded.shape == (2, 16)
        assert not torch.allclose(embedded[0], embedded[1])
        projected_oov = encoder.projection_layer(encoder.oov_embedding.unsqueeze(0)).squeeze(0)
        assert torch.allclose(embedded[1], projected_oov, atol=1e-6)

    def test_oov_with_hashing_strategy(self, feature_meta):
        encoder = CategoricalFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            oov_strategy='hashing'
        )
        x_with_oov = torch.tensor([0, 150])  # 150 >= cardinality
        embedded = encoder(x_with_oov)
        assert embedded.shape == (2, 16)
        assert embedded[0].shape == (16,)
        assert embedded[1].shape == (16,)

    def test_get_output_dim(self, feature_meta):
        encoder = CategoricalFeatureEncoder(
            embedding_dim=32,
            feature_metadata=feature_meta
        )
        assert encoder.get_output_dim() == 32


class TestDateTimeFeatureEncoder:
    """Тесты для DateTimeFeatureEncoder"""

    @pytest.fixture
    def feature_meta(self):
        return FeatureMetadata(
            name="dt_feature",
            dtype="datetime"
        )

    def test_basic_encoding(self, feature_meta):
        encoder = DateTimeFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta
        )
        x = torch.tensor([0.0, 12.0, 23.0])
        embedded = encoder(x)
        assert embedded.shape == (3, 16)

    def test_odd_embedding_dim(self, feature_meta):
        encoder = DateTimeFeatureEncoder(
            embedding_dim=15, # Нечетное
            feature_metadata=feature_meta
        )
        x = torch.tensor([5.0])
        embedded = encoder(x)
        assert embedded.shape == (1, 15)

    def test_get_output_dim(self, feature_meta):
        encoder = DateTimeFeatureEncoder(
            embedding_dim=32,
            feature_metadata=feature_meta
        )
        assert encoder.get_output_dim() == 32


class TestTextFeatureEncoder:

    @pytest.fixture
    def feature_meta(self):
        return FeatureMetadata(
            name="txt_feature",
            dtype="text"
        )

    @patch('src.omniseqbert.models.features_encoders.AutoTokenizer.from_pretrained')
    @patch('src.omniseqbert.models.features_encoders.AutoModel.from_pretrained')
    def test_basic_encoding(self, mock_model, mock_tokenizer, feature_meta):
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones((2, 10), dtype=torch.long)
        }

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 384
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.return_value = MagicMock(last_hidden_state=torch.randn(2, 10, 384))

        encoder = TextFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            model_name='mock-model-name'
        )
        texts = ["Test test test test", "Fuck ML"]
        embedded = encoder(texts)

        mock_tokenizer.assert_called_once_with('mock-model-name')
        mock_model.assert_called_once_with('mock-model-name')
        mock_model_instance.assert_called_once()

        assert embedded.shape == (2, 16)

    @patch('src.omniseqbert.models.features_encoders.AutoTokenizer.from_pretrained')
    @patch('src.omniseqbert.models.features_encoders.AutoModel.from_pretrained')
    def test_pooling_strategies(self, mock_model, mock_tokenizer, feature_meta):
        """Тестирует разные стратегии пулинга."""
        # Настраиваем моки
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 5)),
            'attention_mask': torch.ones((1, 5), dtype=torch.long)
        }

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 384
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.return_value = MagicMock(last_hidden_state=torch.randn(1, 5, 384))

        for strategy in ['mean', 'cls', 'last']:
            encoder = TextFeatureEncoder(
                embedding_dim=16,
                feature_metadata=feature_meta,
                pooling_strategy=strategy
            )
            embedded = encoder(["dummy text"])
            assert embedded.shape == (1, 16)

    @patch('src.omniseqbert.models.features_encoders.AutoTokenizer.from_pretrained')
    @patch('src.omniseqbert.models.features_encoders.AutoModel.from_pretrained')
    def test_invalid_pooling_strategy(self, mock_model, mock_tokenizer, feature_meta):
        """Тестирует ошибку при неверной стратегии пулинга."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 5)),
            'attention_mask': torch.ones((1, 5), dtype=torch.long)
        }

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 384
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.return_value = MagicMock(last_hidden_state=torch.randn(1, 5, 384))

        encoder = TextFeatureEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            pooling_strategy='invalid'
        )
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            encoder(["dummy text"])

    @patch('src.omniseqbert.models.features_encoders.AutoTokenizer.from_pretrained')
    @patch('src.omniseqbert.models.features_encoders.AutoModel.from_pretrained')
    def test_get_output_dim(self, mock_model, mock_tokenizer, feature_meta):
        """Тестирует метод get_output_dim."""
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.config.hidden_size = 384

        encoder = TextFeatureEncoder(
            embedding_dim=32,
            feature_metadata=feature_meta
        )
        assert encoder.get_output_dim() == 32


class TestPretrainedEmbeddingEncoder:
    """Тесты для PretrainedEmbeddingEncoder"""

    @pytest.fixture
    def feature_meta(self):
        return FeatureMetadata(
            name="emb_feature",
            dtype="embedding"
        )

    def test_basic_encoding_same_dim(self, feature_meta):
        encoder = PretrainedEmbeddingEncoder(
            embedding_dim=16,
            feature_metadata=feature_meta,
            input_dim=16
        )
        x = torch.randn(3, 16)
        embedded = encoder(x)
        assert embedded.shape == (3, 16)

    def test_encoding_with_projection(self, feature_meta):
        encoder = PretrainedEmbeddingEncoder(
            embedding_dim=32,
            feature_metadata=feature_meta,
            input_dim=16
        )
        x = torch.randn(2, 16)
        embedded = encoder(x)
        assert embedded.shape == (2, 32)

    def test_get_output_dim(self, feature_meta):
        encoder = PretrainedEmbeddingEncoder(
            embedding_dim=64,
            feature_metadata=feature_meta
        )
        assert encoder.get_output_dim() == 64


class TestEncoderRegistry:
    """Тест реестр и геттер"""

    def test_registry_contents(self):
        expected_keys = {'numerical',
                         'categorical',
                         'datetime',
                         'text',
                         'embedding'}
        assert set(ENCODER_REGISTRY.keys()) == expected_keys

    def test_get_encoder_class_valid(self):
        for enc_type in ENCODER_REGISTRY:
            cls = get_encoder_class(enc_type)
            assert cls == ENCODER_REGISTRY[enc_type]

    def test_get_encoder_class_invalid(self):
        with pytest.raises(ValueError,
                           match="Unknown encoder type: invalid_type"):
            get_encoder_class("invalid_type")


if __name__ == "__main__":
    pytest.main([__file__])
