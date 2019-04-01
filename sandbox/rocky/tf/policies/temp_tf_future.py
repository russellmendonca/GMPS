

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.contrib.layers.python.layers import utils

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'

def spatial_softmax(features,
                    temperature=None,
                    name=None,
                    variables_collections=None,
                    trainable=True,
                    data_format='NHWC'):
  """Computes the spatial softmax of a convolutional feature map.
  First computes the softmax over the spatial extent of each channel of a
  convolutional feature map. Then computes the expected 2D position of the
  points of maximal activation for each channel, resulting in a set of
  feature keypoints [x1, y1, ... xN, yN] for all N channels.
  Read more here:
  "Learning visual feature spaces for robotic manipulation with
  deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
  Args:
    features: A `Tensor` of size [batch_size, W, H, num_channels]; the
      convolutional feature map.
    temperature: Softmax temperature (optional). If None, a learnable
      temperature is created.
    name: A name for this operation (optional).
    variables_collections: Collections for the temperature variable.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
  Returns:
    feature_keypoints: A `Tensor` with size [batch_size, num_channels * 2];
      the expected 2D locations of each channel's feature keypoint (normalized
      to the range (-1,1)). The inner dimension is arranged as
      [x1, y1, ... xN, yN].
  Raises:
    ValueError: If unexpected data_format specified.
    ValueError: If num_channels dimension is unspecified.
  """
  with tf.variable_scope(name, 'spatial_softmax'):
    shape = array_ops.shape(features)
    
    static_shape = features.shape
    if data_format == DATA_FORMAT_NHWC:
      height, width, num_channels = shape[1], shape[2], static_shape[3]
    elif data_format == DATA_FORMAT_NCHW:
      num_channels, height, width = static_shape[1], shape[2], shape[3]
    else:
      raise ValueError('data_format has to be either NCHW or NHWC.')
    if num_channels.value is None:
      raise ValueError('The num_channels dimension of the inputs to '
                       '`spatial_softmax` should be defined. Found `None`.')

    with ops.name_scope('spatial_softmax_op', 'spatial_softmax_op', [features]):
      # Create tensors for x and y coordinate values, scaled to range [-1, 1].
      pos_x, pos_y = array_ops.meshgrid(
          math_ops.lin_space(-1., 1., num=height),
          math_ops.lin_space(-1., 1., num=width),
          indexing='ij')
      pos_x = array_ops.reshape(pos_x, [height * width])
      pos_y = array_ops.reshape(pos_y, [height * width])

      if temperature is None:
        temp_initializer = init_ops.ones_initializer()
      else:
        temp_initializer = init_ops.constant_initializer(temperature)

      if not trainable:
        temp_collections = None
      else:
        temp_collections = utils.get_variable_collections(
            variables_collections, 'temperature')

      temperature = variables.model_variable(
          'temperature',
          shape=(),
          dtype=dtypes.float32,
          initializer=temp_initializer,
          collections=temp_collections,
          trainable=trainable)
      if data_format == 'NCHW':
        features = array_ops.reshape(features, [-1, height * width])
      else:
        features = array_ops.reshape(
            array_ops.transpose(features, [0, 3, 1, 2]), [-1, height * width])

      softmax_attention = nn.softmax(features / temperature)
      expected_x = math_ops.reduce_sum(
          pos_x * softmax_attention, [1], keep_dims=True)
      expected_y = math_ops.reduce_sum(
          pos_y * softmax_attention, [1], keep_dims=True)
      expected_xy = array_ops.concat([expected_x, expected_y], 1)
      feature_keypoints = array_ops.reshape(expected_xy,
                                            [-1, num_channels.value * 2])
      feature_keypoints.set_shape([None, num_channels.value * 2])
  return feature_keypoints
