from typing import Literal

FeatureType = Literal['numerical', 'categorical',
                      'binary', 'datetime', 'text',
                      'embedding', 'custom']
PositionEncodingType = Literal['absolute', 'relative',
                               'temporal', 'hybrid', 'none']
ActivationType = Literal['gelu', 'relu', 'swish', 'tanh', 'sigmoid']
TaskType = Literal['masked_recovery', 'next_value',
                   'classification', 'regression']
