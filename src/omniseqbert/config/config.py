import json
import yaml

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, get_args
from dataclasses import dataclass, field, fields

from .types import FeatureType, PositionEncodingType, ActivationType, TaskType


def _get_literal_values(literal_type) -> List[str]:
    return list(get_args(literal_type))


def _validate_literal(value: str, literal_type, param_name: str):
    allowed_values = _get_literal_values(literal_type)
    if value not in allowed_values:
        raise ValueError(f"allowed: {allowed_values}")


def validate_config_types(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, f.type):
                expected_type = f.type
                raise TypeError(f"Field '{f.name}' \
                                expected type '{expected_type}', \
                                got '{type(value)}'.")

            # проверка для Literal
            if hasattr(expected_type,
                       "__origin__") and expected_type.__origin__ is Literal:
                _validate_literal(value, expected_type, f.name)

    cls.__init__ = new_init
    return cls


@validate_config_types
@dataclass
class FeatureEncoderConfig:
    default_embedding_dim: int = 64
    fusion_method: Literal['concat', 'attention', 'gating'] = 'attention'
    dropout_rate: float = 0.1

    numerical: Dict[str, Any] = field(default_factory=lambda: {
        'normalization': 'adaptive',
        # 'adaptive', 'min_max', 'standard'
        'missing_value_strategy': 'special_token'
        # 'special_token', 'interpolate', 'drop'
    })
    categorical: Dict[str, Any] = field(default_factory=lambda: {
        'max_cardinality': 10000,
        'oov_strategy': 'hashing',
        # 'hashing', 'token'
        'embedding_dim_ratio': 0.5
        # ratio for calculating embedding dim from cardinality
    })
    text: Dict[str, Any] = field(default_factory=lambda: {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'max_length': 32
        # это базовые настройки которые я использую
        # TODO: добавить селектер для sklearn методов или самописных эмбеддеров
    })


@validate_config_types
@dataclass
class PositionEncoderConfig:
    type: PositionEncodingType = 'hybrid'
    max_sequence_length: int = 512
    max_relative_distance: int = 64
    dropout_rate: float = 0.1

    temporal: Dict[str, Any] = field(default_factory=lambda: {
        'time_unit': 'seconds',
        # 'seconds', 'minutes', 'hours', 'days'
        'cyclic_features': True
    })


@validate_config_types
@dataclass
class TransformerConfig:
    num_layers: int = 6
    num_heads: int = 8
    hidden_size: int = 384
    intermediate_size: Optional[int] = None  # 4 * hidden_size ниже
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    activation: ActivationType = 'gelu'
    use_flash_attention: bool = True

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


@validate_config_types
@dataclass
class TaskHeadConfig:
    enabled: List[TaskType] = field(
        default_factory=lambda: ['masked_recovery', 'next_value']
        )
    masked_recovery: Dict[str, Any] = field(default_factory=lambda: {
        'loss_weights': {
            'numerical': 1.0,
            'categorical': 2.0,
            'text': 1.5
        },
        'mask_probability': 0.15,
        'mask_strategies': {
            'token_mask': 0.8,
            'random_token': 0.1,
            'unchanged': 0.1
        }
    })
    next_value: Dict[str, Any] = field(default_factory=lambda: {
        'prediction_horizon': 1,
        'loss_function': 'mse'
        # 'mse', 'mae', 'quantile'
    })


@validate_config_types
@dataclass
class TrainingConfig:
    objective: Literal['cloze', 'next_value', 'hybrid'] = 'cloze'
    batch_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    warmup_steps: int = 1000
    max_epochs: int = 100
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


@validate_config_types
@dataclass
class DataConfig:
    sequence_length: int = 128
    stride: int = 64
    min_sequence_length: int = 5
    max_missing_values: float = 0.5
    batch_construction: Literal['length_bucketing',
                                'random'] = 'length_bucketing'
    num_workers: int = 8
    pin_memory: bool = True
    streaming: bool = False


@validate_config_types
@dataclass
class OptimizationsConfig:
    mixed_precision: bool = True
    distributed: bool = True
    gradient_checkpointing: bool = True
    flash_attention: bool = True


@validate_config_types
@dataclass
class OmniSeqBERTConfig:
    model_name: str = "omniseqbert-base"  # cloze задача

    features: Dict[str, FeatureType] = field(default_factory=dict)
    # {'column_name': 'type'}

    feature_encoder: FeatureEncoderConfig = field(
        default_factory=FeatureEncoderConfig)

    position_encoder: PositionEncoderConfig = field(
        default_factory=PositionEncoderConfig)

    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    task_heads: TaskHeadConfig = field(default_factory=TaskHeadConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)

    data: DataConfig = field(default_factory=DataConfig)

    optimizations: OptimizationsConfig = field(
        default_factory=OptimizationsConfig)

    seed: int = 42  # 42 братуха
    device: str = "cuda"
    log_level: str = "info"
    experiment_name: str = "default_experiment"
    auto_infer_types: bool = True

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, 'to_dict'):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OmniSeqBERTConfig':
        nested_configs = {}
        for f in fields(cls):
            if f.name in config_dict and hasattr(f.type, '__annotations__'):
                nested_cls = f.type  # dataclass
                if hasattr(nested_cls, 'from_dict'):
                     nested_configs[f.name] = nested_cls.from_dict(
                         config_dict[f.name])
                else:
                    nested_configs[f.name] = nested_cls(**config_dict[f.name])

        init_dict = {**config_dict, **nested_configs}
        return cls(**init_dict)

    def to_json(self, file_path: Union[str, Path]):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'OmniSeqBERTConfig':
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, file_path: Union[str, Path]):
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f,
                      default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'OmniSeqBERTConfig':
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
