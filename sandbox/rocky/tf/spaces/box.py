


from rllab.spaces.box import Box as TheanoBox
import tensorflow as tf


class Box(TheanoBox):
    def new_tensor_variable(self, name, extra_dims, add_to_flat_dim=0):
        return tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.flat_dim+add_to_flat_dim], name=name)
