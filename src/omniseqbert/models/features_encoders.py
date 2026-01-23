from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from dataclasses import dataclass


@dataclass
class FeatureMetadata:
    name: str
    dtype: str
    # 'numerical', 'categorical', 'binary', 'datetime', 'text', 'embedding'
    cardinality: Optional[int] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None


class BaseFeatureEncoder(nn.Module, ABC):
    def __init__(self, embedding_dim: int, feature_metadata: FeatureMetadata):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_metadata = feature_metadata

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        pass


class NumericalFeatureEncoder(BaseFeatureEncoder):
    def __init__(
        self,
        embedding_dim: int,
        feature_metadata: FeatureMetadata,
        normalization: str = 'adaptive',
        missing_value_strategy: str = 'special_token'
    ):
        super().__init__(embedding_dim, feature_metadata)
        self.normalization = normalization
        self.missing_value_strategy = missing_value_strategy

        self.register_buffer('_min_val', torch.tensor(feature_metadata.min_val or 0.0))
        self.register_buffer('_max_val', torch.tensor(feature_metadata.max_val or 1.0))

        self.projection_layer = nn.Linear(1, embedding_dim)

        if missing_value_strategy == 'special_token':
            self.missing_token_embedding = nn.Parameter(torch.randn(embedding_dim))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == 'min_max':
            return (x - self._min_val) / (self._max_val - self._min_val + 1e-8)
        elif self.normalization == 'adaptive':
            range_val = self._max_val - self._min_val
            if range_val > 1000:
                x_log = torch.log(torch.abs(x - self._min_val) + 1)
                max_log = torch.log(range_val + 1)
                normalized = x_log / max_log
            else:
                normalized = (x - self._min_val) / (range_val + 1e-8)
            return torch.clamp(normalized, 0.0, 1.0)
        else:
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        missing_mask = torch.isnan(x)
        x_norm = self._normalize(x)
        x_expanded = x_norm.unsqueeze(-1)
        embedded = self.projection_layer(x_expanded)

        if self.missing_value_strategy == 'special_token':
            embedded = torch.where(
                missing_mask.unsqueeze(-1),
                self.missing_token_embedding.expand_as(embedded),
                embedded
            )
        return embedded

    def get_output_dim(self) -> int:
        return self.embedding_dim


class CategoricalFeatureEncoder(BaseFeatureEncoder):
    def __init__(
        self,
        embedding_dim: int,  # hidden
        feature_metadata: FeatureMetadata,
        max_cardinality: int = 10000,
        oov_strategy: str = 'hashing',
    ):
        super().__init__(embedding_dim, feature_metadata)
        self.max_cardinality = max_cardinality
        self.oov_strategy = oov_strategy
        self.cat_embedding_dim = embedding_dim

        actual_cardinality = min(max_cardinality,
                                 feature_metadata.cardinality or 100)
        self.embedding_table = nn.Embedding(actual_cardinality,
                                            self.cat_embedding_dim)

        if oov_strategy == 'token':
            self.oov_embedding = nn.Parameter(torch.randn(self.cat_embedding_dim))
        else:
            self.oov_embedding = None

        # self.projection_layer = nn.Linear(self.cat_embedding_dim, embedding_dim)
        # self.projection_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        oov_mask = x >= self.embedding_table.num_embeddings
        x_safe = torch.where(oov_mask, torch.zeros_like(x), x)
        embedded_cat = self.embedding_table(x_safe)
        # (..., cat_dim) = (..., dim)

        if self.oov_strategy == 'token' and self.oov_embedding is not None:
            embedded_cat = torch.where(
                oov_mask.unsqueeze(-1),
                self.oov_embedding.expand_as(embedded_cat),
                embedded_cat
            )

        # embedded_out = self.projection_layer(embedded_cat)
        return embedded_cat

    def get_output_dim(self) -> int:
        return self.embedding_dim
    # cat_embedding_dim == embedding_dim


class DateTimeFeatureEncoder(BaseFeatureEncoder):
    def __init__(
        self,
        embedding_dim: int,
        feature_metadata: FeatureMetadata,
    ):
        super().__init__(embedding_dim, feature_metadata)
        self.num_cyclical_features = 4
        self.projection_layer = nn.Linear(self.num_cyclical_features,
                                          embedding_dim)

    def _get_time_features(self,
                           timestamps: torch.Tensor) -> Dict[str,
                                                             torch.Tensor]:
        hours = timestamps % 24
        days_of_week = (timestamps / 24) % 7
        return {
            'hours_sin': torch.sin(2 * np.pi * hours / 24),
            'hours_cos': torch.cos(2 * np.pi * hours / 24),
            'days_sin': torch.sin(2 * np.pi * days_of_week / 7),
            'days_cos': torch.cos(2 * np.pi * days_of_week / 7),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._get_time_features(x)
        combined_features = torch.stack([
            features['hours_sin'],
            features['hours_cos'],
            features['days_sin'],
            features['days_cos']
        ], dim=-1)
        embedded = self.projection_layer(combined_features)
        return embedded

    def get_output_dim(self) -> int:
        return self.embedding_dim


class TextFeatureEncoder(BaseFeatureEncoder):
    def __init__(
        self,
        embedding_dim: int,
        feature_metadata: FeatureMetadata,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        max_length: int = 32,
        pooling_strategy: str = 'mean'
    ):
        super().__init__(embedding_dim, feature_metadata)
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder_model = AutoModel.from_pretrained(model_name)
        self.pretrained_dim = self.encoder_model.config.hidden_size
        self.projection_layer = nn.Linear(self.pretrained_dim, embedding_dim)

    def forward(self, x: List[str]) -> torch.Tensor:
        """
        param x: List[str] с текстами
        return: torch.Tensor (len(x), dim) это типо шейп
        """
        inputs = self.tokenizer(
            x,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        device = inputs['input_ids'].device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state

        if self.pooling_strategy == 'mean':
            embedded_text = last_hidden_states.mean(dim=1)
        elif self.pooling_strategy == 'cls':
            embedded_text = last_hidden_states[:, 0, :]
        elif self.pooling_strategy == 'last':
            embedded_text = last_hidden_states[:, -1, :]
        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.pooling_strategy}")

        embedded_out = self.projection_layer(embedded_text)
        return embedded_out

    def get_output_dim(self) -> int:
        return self.embedding_dim


class PretrainedEmbeddingEncoder(BaseFeatureEncoder):
    def __init__(
        self,
        embedding_dim: int,
        feature_metadata: FeatureMetadata,
        input_dim: Optional[int] = None
    ):
        super().__init__(embedding_dim, feature_metadata)
        self.input_dim = input_dim or embedding_dim
        if self.input_dim != embedding_dim:
            self.projection_layer = nn.Linear(self.input_dim, embedding_dim)
        else:
            self.projection_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)

    def get_output_dim(self) -> int:
        return self.embedding_dim


ENCODER_REGISTRY = {
    'numerical': NumericalFeatureEncoder,
    'categorical': CategoricalFeatureEncoder,
    'datetime': DateTimeFeatureEncoder,
    'text': TextFeatureEncoder,
    'embedding': PretrainedEmbeddingEncoder,
}


def get_encoder_class(encoder_type: str) -> type:
    encoder_class = ENCODER_REGISTRY.get(encoder_type)
    if encoder_class is None:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(ENCODER_REGISTRY.keys())}")
    return encoder_class


__all__ = [
    'FeatureMetadata',
    'BaseFeatureEncoder',
    'NumericalFeatureEncoder',
    'CategoricalFeatureEncoder',
    'DateTimeFeatureEncoder',
    'TextFeatureEncoder',
    'PretrainedEmbeddingEncoder',
    'ENCODER_REGISTRY',
    'get_encoder_class'
]
