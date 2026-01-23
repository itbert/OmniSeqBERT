from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .features_encoders import ENCODER_REGISTRY, FeatureMetadata
import os

# TODO: пофиксить варнинг с размерностями тензора таргета
# self.hidden_dim.reshape[n, ] (??????)


class OmniSeqBERT(nn.Module):
    """
    Обобщённая модель
    с поддержкой разнородных признаков на каждом шаге последовательности.
    Может работать с любым набором признаков
    categorical, numerical, datetime, text, embedding, других пока нету
    Поддерживает и основана на задаче cloze
    """

    def __init__(
        self,
        feature_configs: Dict[str, FeatureMetadata],
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout_prob: float = 0.1,
        max_seq_len: int = 200,
        fusion_method: str = 'sum',  # 'sum' | 'concat_proj'
        maskable_features: Optional[List[str]] = None,
    ):
        """
        Args:
            feature_configs: dict/json/yaml {feature_name: FeatureMetadata}
            {"item_id": FeatureMetadata(...), "price": ...} как инпут
            hidden_dim: Размерность скрытого слоя
            num_layers: Количество Transformerов
            num_heads: Количество голов в multi-head attention
            dropout_prob: Вероятность dropout
            max_seq_len: Максимальная длина последовательности
            fusion_method: Как объединять эмбеддингов ('sum' или 'concat_proj')
            maskable_features:  Список имён признаков,
                                которые можно маскировать и восстанавливать
                                Если None, то все признаки из feature_configs
                                                        считаются маскируемыми
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.fusion_method = fusion_method
        self.maskable_features = maskable_features or list(feature_configs.keys())

        self.encoders = nn.ModuleDict()
        for name, meta in feature_configs.items():
            enc_cls = ENCODER_REGISTRY[meta.dtype]
            self.encoders[name] = enc_cls(embedding_dim=hidden_dim,
                                          feature_metadata=meta)

        if fusion_method == 'concat_proj':
            total_input_dim = len(feature_configs) * hidden_dim
            self.fusion_layer = nn.Linear(total_input_dim, hidden_dim)
        elif fusion_method == 'sum':
            self.fusion_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown method: {fusion_method}. Use 'sum' or 'concat_proj'")
        self.fusion_dropout = nn.Dropout(dropout_prob)

        self.pos_encoding = nn.Embedding(max_seq_len, hidden_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout_prob,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.reconstruction_heads = nn.ModuleDict()
        for name in self.maskable_features:
            if name in self.encoders:
                meta = feature_configs[name]
                head = self._create_reconstruction_head(name, meta)
                if head is not None:
                    self.reconstruction_heads[name] = head
                else:
                    print(f"Warning: Could not create reconstruction head for feature '{name}'. Skipping.")
            else:
                print(f"Warning: Feature '{name}' marked as maskable but no encoder found. Skipping.")

        self.apply(self._init_weights)

    def _create_reconstruction_head(self,
                                    name: str,
                                    meta: FeatureMetadata
                                    ) -> Optional[nn.Module]:
        """head для восстановления конкретного признака"""
        if meta.dtype == 'categorical':
            enc = self.encoders[name]
            if hasattr(enc, 'embedding_table'):
                vocab_size = enc.embedding_table.num_embeddings
                return nn.Linear(self.hidden_dim, vocab_size)
            else:
                vocab_size = meta.cardinality or 1000
                return nn.Linear(self.hidden_dim, vocab_size)
        elif meta.dtype == 'numerical':
            return nn.Linear(self.hidden_dim, 1)
        elif meta.dtype == 'datetime':
            return nn.Linear(self.hidden_dim, 4)
        elif meta.dtype == 'text':
            return nn.Linear(self.hidden_dim, self.hidden_dim)
        elif meta.dtype == 'embedding':
            enc = self.encoders[name]
            input_dim = getattr(enc, 'input_dim', self.hidden_dim)
            return nn.Linear(self.hidden_dim, input_dim)
        else:
            return nn.Linear(self.hidden_dim, self.hidden_dim)

    def _init_weights(self, module):
        """Инициализация весов"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                features_batch: Dict[str, torch.Tensor],
                pad_mask: Optional[torch.Tensor] = None):
        """
        Args:
            features_batch:
                dict/json/yaml {feature_name: tensor(batch_size, seq_len, ...)}
                Зависит от типа признака (batch_size, seq_len) для скаляров
            pad_mask: Bool tensor (batch_size, seq_len)
                      True для реальных токенов, False для паддинга
        Returns:
            last_hidden_states: torch.Tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len = next(iter(features_batch.values())).shape[:2]

        embeddings_list = []
        for name, x in features_batch.items():
            if name in self.encoders:
                emb = self.encoders[name](x)
                embeddings_list.append(emb)
            else:
                raise KeyError(f"Feature '{name}' not found in initialized encoders.")

        if self.fusion_method == 'sum':
            combined_emb = torch.stack(embeddings_list, dim=0).sum(dim=0)
        elif self.fusion_method == 'concat_proj':
            concatenated_emb = torch.cat(embeddings_list, dim=-1)
            combined_emb = self.fusion_layer(concatenated_emb)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        combined_emb = self.fusion_dropout(combined_emb)

        positions = torch.arange(
            seq_len,
            device=combined_emb.device
            ).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_encoding(positions)
        final_emb = combined_emb + pos_emb

        attn_mask = None
        if pad_mask is not None:
            attn_mask = ~pad_mask.bool()

        hidden_states = final_emb
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states,
                                  src_key_padding_mask=attn_mask)

        last_hidden_states = self.final_layer_norm(hidden_states)

        return last_hidden_states

    def reconstruct_features(self,
                             last_hidden_states: torch.Tensor,
                             masked_positions: torch.Tensor
                             ) -> Dict[str, torch.Tensor]:
        """
        Args:
            last_hidden_states: torch.Tensor (batch_size, seq_len, hidden_dim)
            masked_positions: Bool tensor (batch_size, seq_len)
                              True на позициях для предсказания
        Returns:
            reconstructed_features:
                any2dict {feature_name: predicted_tensor}
                predicted_tensor может быть (total_masked, output_dim_feature)
        """
        selected_states = last_hidden_states[masked_positions]

        reconstructed = {}
        for name, head in self.reconstruction_heads.items():
            pred = head(selected_states)  # (total_masked, output_dim_feature)
            reconstructed[name] = pred

        return reconstructed

    def calculate_loss(self,
                       reconstructed_features: Dict[str, torch.Tensor],
                       original_batch: Dict[str, torch.Tensor],
                       masked_positions: torch.Tensor) -> float:
        """
        Loss для всех восстанавливаемых признаков на замаскированных позициях
        Cloze task objective
        """
        total_loss = 0.0
        for name, pred_logits_or_values in reconstructed_features.items():
            if name not in original_batch:
                continue

            original_values = original_batch[name][masked_positions]
            # (total_masked, ...)

            meta = self.encoders[name].feature_metadata
            if meta.dtype == 'categorical':
                target = original_values.long()
                criterion = nn.CrossEntropyLoss(reduction='mean')
                loss = criterion(pred_logits_or_values, target)
            elif meta.dtype in ['numerical', 'datetime', 'embedding', 'text']:
                target = original_values.float()
                criterion = nn.MSELoss(reduction='mean')
                loss = criterion(pred_logits_or_values, target)
            else:
                target = original_values.float()
                criterion = nn.MSELoss(reduction='mean')
                loss = criterion(pred_logits_or_values, target)

            total_loss += loss

        return total_loss

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory,
                                                   "omniseqbert-base.bin"))

    @classmethod
    def from_pretrained(cls, save_directory: str,
                        feature_configs: Dict[str, FeatureMetadata],
                        task_type: str = 'cpu',
                        **kwargs):
        model = cls(feature_configs=feature_configs, **kwargs)
        state_dict_path = os.path.join(save_directory, "omniseqbert-base.bin")
        model.load_state_dict(torch.load(state_dict_path,
                                         map_location=torch.device(task_type)))
        # TODO: fix allocate
        return model

    def get_item_embeddings(self,
                            item_feature_name: str = 'item_id'
                            ) -> Optional[torch.Tensor]:
        """
        Возвращает эмбеддинги item
        Если dim(embedding_table) == hidden_dim модели то выдаст набор векторов
        """
        if item_feature_name in self.encoders:
            enc = self.encoders[item_feature_name]
            if enc.feature_metadata.dtype == 'categorical' and hasattr(enc, 'embedding_table'):
                if enc.embedding_table.embedding_dim == self.hidden_dim:
                    return enc.embedding_table.weight
                else:
                    print(f"Warning: Embedding table for '{item_feature_name}' has dim {enc.embedding_table.embedding_dim} != {self.hidden_dim}. Cannot return embeddings.")
                    return None
        return None

    def predict_next_item(self,
                          features_batch: Dict[str, torch.Tensor],
                          pad_mask: Optional[torch.Tensor] = None):
        """
        Предсказывает следующий item (последняя позиция).
        Использует специальный токен маски, как в оригинальном BERT4Rec.
        Этот метод требует, чтобы 'item_id' был маскируемым и категориальным.
        """
        batch_size, seq_len = next(iter(features_batch.values())).shape[:2]

        item_feature_name = 'item_id'
        if item_feature_name not in self.reconstruction_heads or \
           self.encoders[item_feature_name].feature_metadata.dtype != 'categorical':
            raise RuntimeError(
                f"'{item_feature_name}' must be a maskable categorical feature to predict next item."
            )

        last_hidden_states = self.forward(features_batch, pad_mask)

        last_hs_for_prediction = last_hidden_states[:, -1, :]

        item_logits = self.reconstruction_heads[
            item_feature_name](last_hs_for_prediction)

        return item_logits
