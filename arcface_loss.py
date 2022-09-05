# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ArcFace losses  base class.

ArcFace: Additive Angular Margin Loss for Deep Face
Recognition. [online] arXiv.org. Available at:
<https://arxiv.org/abs/1801.07698v3>.

"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import tensorflow as tf

from tensorflow_similarity.utils import is_tensor_or_variable
from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor, IntTensor

from .metric_loss import MetricLoss
from .utils import logsumexp
from typing import Any, Callable, Union


def arcface_loss(
        query_labels: IntTensor,
        query_embeddings: FloatTensor,  # output
        key_labels: IntTensor,
        key_embeddings: FloatTensor,  # input
        num_classes: int,
        margin: int,
        scale: float,
        kernel: FloatTensor) -> Any:
    # label
    # eager mode / grass execution
    # cross batch memory
    batch_size = tf.shape(query_labels)[0]

    #embeddingsize * num classess

    query_embeddings = tf.nn.l2_normalize(query_embeddings, axis=1)
    kernel_norm = tf.nn.l2_normalize(kernel, 0)

    cos_theta = tf.matmul(query_embeddings, kernel_norm)
    print("cos_theta shape : ",  cos_theta.shape)


    # the degree matrix normalization
    cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)

    index = tf.where(num_classes != -1).shape[0]

    m_hot = tf.zeros([batch_size, num_classes])
    margin_t = tf.ones([batch_size, num_classes]) * margin

    # there are some issues to update query embeddings according to margin and index
    groups = tf.cast(query_labels[index, None], tf.int32)


    m_hot = tf.scatter_nd(groups, query_embeddings, margin_t)
    # tf.tensor_scatter_nd_add(m_hot, groups, margin_t)

    cos_theta = tf.acos(cos_theta)
    cos_theta = cos_theta.assign_add(tf.where(index, m_hot, tf.zeros_like(m_hot)))
    cos_theta = cos_theta.cos()
    cos_theta = tf.math.multiply(cos_theta, scale)

    loss = tf.keras.crossentropy(cos_theta, query_labels)

    return loss



# l2_normalizaiton tensorflow use

@tf.keras.utils.register_keras_serializable(package="Similarity")
class ArcFaceLoss(MetricLoss):
    """Implement of large margin arc distance:
            Args:
                num_classes: The number of classes in your training dataset
                margin: m in the paper, the angular margin penalty in radians
                scale: s in the paper, feature scale
                regularizer: weights regularizer
            """

    def __init__(self,
                 num_classes: int,
                 margin: float = 0.50,  # margin in radians
                 scale: float = 64.0,  # feature scale
                 name: str = "ArcFaceLoss",
                 reduction: Callable = tf.keras.losses.Reduction.AUTO,
                 **kwargs: object) -> object:
        # distance canonicalization
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.name = name

        super().__init__(
            fn=arcface_loss,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
            reduction=reduction,
            name=name,
            **kwargs
        )


("\n"
 "    def call(self, y_true: FloatTensor, y_pred: FloatTensor) -> FloatTensor:\n"
 "\n"
 "        projector = tf.math.l2_normalize(y_true, axis=1, epsilon=1e-12)\n"
 "        predictor = tf.math.l2_normalize(y_pred, axis=1, epsilon=1e-12)\n"
 "\n"
 "        cos_theta = tf.matmul(projector, predictor)\n"
 "        cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)\n"
 "\n"
 "        index = tf.where(self.num_classes != -1)[0]\n"
 "\n"
 "        m_hot = tf.zeros(index.size()[0], cos_theta.size()[1])\n"
 "\n"
 "        m_hot.scatter_(1, self.num_classes[index, None], self.margin)\n"
 "\n"
 "        cos_theta = tf.acos(cos_theta)\n"
 "        cos_theta[index] += m_hot\n"
 "        cos_theta = cos_theta.cos()\n"
 "        cos_theta = tf.math.multiply(cos_theta, self.scale)\n"
 "\n"
 "        loss: FloatTensor = tf.nn.softmax(cos_theta, dim=1)\n"
 "\n"
 "        return loss\n"
 "\n"
 "    def get_config(self) -> Dict[str, Any]:\n"
 "        config = {\n"
 "            \"name\": self.name,\n"
 "            \"num_classes\": self.num_classes,\n"
 "            \"scale\": self.scale,\n"
 "            \"margin\": self.margin,\n"
 "        }\n"
 "        base_config = super().get_config()\n"
 "        return {**base_config, **config}\n")