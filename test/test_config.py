import pytest
import tempfile
import os
from src.omniseqbert.config.config import (
    OmniSeqBERTConfig,
    FeatureEncoderConfig,
    PositionEncoderConfig,
    TransformerConfig,
    TaskHeadConfig,
    TrainingConfig,
    DataConfig,
    OptimizationsConfig
)


class TestOmniSeqBERTConfig:
    def test_default_initialization(self):
        config = OmniSeqBERTConfig()

        assert config.model_name == "omniseqbert-base"
        assert config.seed == 42
        assert config.device == "cuda"
        assert isinstance(config.feature_encoder, FeatureEncoderConfig)
        assert isinstance(config.position_encoder, PositionEncoderConfig)
        assert isinstance(config.transformer, TransformerConfig)
        assert isinstance(config.task_heads, TaskHeadConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.optimizations, OptimizationsConfig)

    def test_literal_validation_on_init(self):
        config = OmniSeqBERTConfig(device="cpu")
        assert config.device == "cpu"

        with pytest.raises(ValueError, match="'invalid_device' is not a valid value for 'device'"):
            OmniSeqBERTConfig(device="invalid_device")

    def test_nested_config_initialization(self):
        nested_transformer = TransformerConfig(
            num_layers=4,
            num_heads=6,
            hidden_size=256,
            activation='relu'
        )
        config = OmniSeqBERTConfig(transformer=nested_transformer)

        assert config.transformer.num_layers == 4
        assert config.transformer.activation == 'relu'

    def test_to_from_dict(self):
        original_config = OmniSeqBERTConfig(
            model_name="test_model",
            seed=123,
            feature_encoder=FeatureEncoderConfig(default_embedding_dim=128)
        )

        config_dict = original_config.to_dict()

        assert 'feature_encoder' in config_dict
        assert isinstance(config_dict['feature_encoder'], dict)
        assert config_dict['feature_encoder']['default_embedding_dim'] == 128

        restored_config = OmniSeqBERTConfig.from_dict(config_dict)

        assert restored_config.model_name == "test_model"
        assert restored_config.seed == 123
        assert restored_config.feature_encoder.default_embedding_dim == 128

    def test_to_from_json(self):
        original_config = OmniSeqBERTConfig(model_name="json_test_model")

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', 
                                         delete=False) as tmp_file:
            temp_filename = tmp_file.name

        try:
            original_config.to_json(temp_filename)
            restored_config = OmniSeqBERTConfig.from_json(temp_filename)

            assert restored_config.model_name == "json_test_model"
        finally:
            os.unlink(temp_filename)

    def test_to_from_yaml(self):
        original_config = OmniSeqBERTConfig(model_name="yaml_test_model")

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.yaml', delete=False) as tmp_file:
            temp_filename = tmp_file.name

        try:
            original_config.to_yaml(temp_filename)
            restored_config = OmniSeqBERTConfig.from_yaml(temp_filename)

            assert restored_config.model_name == "yaml_test_model"
        finally:
            os.unlink(temp_filename)


class TestFeatureEncoderConfig:
    def test_default_values(self):
        config = FeatureEncoderConfig()
        assert config.default_embedding_dim == 64
        assert config.fusion_method == 'attention'
        assert config.dropout_rate == 0.1

    def test_literal_validation(self):
        config = FeatureEncoderConfig(fusion_method='concat')
        assert config.fusion_method == 'concat'

        with pytest.raises(ValueError, match="'invalid_fusion' is not a valid value for 'fusion_method'"):
            FeatureEncoderConfig(fusion_method='invalid_fusion')


class TestPositionEncoderConfig:
    def test_literal_validation(self):
        config = PositionEncoderConfig(type='relative')
        assert config.type == 'relative'

        with pytest.raises(ValueError, match="'invalid_pos_type' is not a valid value for 'type'"):
            PositionEncoderConfig(type='invalid_pos_type')


class TestTransformerConfig:
    def test_intermediate_size_calculation(self):
        config = TransformerConfig(hidden_size=256, intermediate_size=None)
        assert config.intermediate_size == 4 * 256  # 1024

        config_with_explicit_intermediate = TransformerConfig(
            hidden_size=256,
            intermediate_size=512
        )
        assert config_with_explicit_intermediate.intermediate_size == 512

    def test_literal_validation(self):
        config = TransformerConfig(activation='swish')
        assert config.activation == 'swish'

        with pytest.raises(ValueError, match="'invalid_act' is not a valid value for 'activation'"):
            TransformerConfig(activation='invalid_act')


class TestTaskHeadConfig:
    def test_enabled_tasks_validation(self):
        config = TaskHeadConfig(enabled=['next_value', 'classification'])
        assert 'next_value' in config.enabled

        with pytest.raises(ValueError, match="'invalid_task' is not a valid value for 'enabled\\[item\\]'"):
            TaskHeadConfig(enabled=['invalid_task'])


class TestDataConfig:
    def test_literal_validation(self):
        config = DataConfig(batch_construction='random')
        assert config.batch_construction == 'random'

        with pytest.raises(ValueError, match="'invalid_batch' is not a valid value for 'batch_construction'"):
            DataConfig(batch_construction='invalid_batch')


def test_validate_config_types_decorator():
    from src.omniseqbert.config.config import validate_config_types
    from dataclasses import dataclass
    from typing import Literal

    @validate_config_types
    @dataclass
    class TestConfig:
        name: str
        count: int
        activation: Literal['relu', 'tanh']

    config = TestConfig(name="test", count=5, activation='relu')
    assert config.name == "test"
    assert config.count == 5
    assert config.activation == 'relu'

    with pytest.raises(TypeError, match="Field 'count' expected type '.*int.*', got '.*str.*'"):
        TestConfig(name="test", count="not_an_int", activation='relu')

    with pytest.raises(ValueError, match="'invalid_act' is not a valid value for 'activation'"):
        TestConfig(name="test", count=5, activation='invalid_act')


if __name__ == "__main__":
    pytest.main([__file__])
