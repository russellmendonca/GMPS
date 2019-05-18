import numpy as np
import scipy.optimize
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils
from collections import OrderedDict




class QuadDistExpertOptimizer(Serializable):
    """
    Runs Tensorflow optimization on a quadratic loss function

    """

    def __init__(
            self,
            name,
            max_opt_itr=20,
            initial_penalty=1.0,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            max_penalty_itr=10,
            adapt_penalty=True,
            adam_steps=5,
            use_momentum_optimizer=False,
    ):
        Serializable.quick_init(self, locals())
        self._name = name
        assert len([var for var in tf.global_variables() if self._name in var.name]) == 0, "please choose a different name for your optimizer"

        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._max_penalty_itr = max_penalty_itr
        self._adapt_penalty = adapt_penalty

        self._opt_fun = None
        self._target = None
        self._optimizer_steps = adam_steps
        self._correction_term = None
        self._use_momentum_optimizer=use_momentum_optimizer


    def update_opt(self, loss, target,  inputs,  *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        self._target = target
        self._inputs = inputs
        self._loss = loss
       
        if self._use_momentum_optimizer:
            self._adam=tf.train.MomentumOptimizer(learning_rate=0.0000001, momentum=0.997, name=self._name)
            assert False, "not supported at the moment"
        else:
            self._adam = tf.train.AdamOptimizer(name=self._name) #learning_rate=0.001
        self._optimizer_vars_initializers = [var.initializer for var in tf.global_variables() if self._name in var.name]

        self._gradients = self._adam.compute_gradients(loss=self._loss, var_list=self._target)
        self._train_step = self._adam.apply_gradients(self._gradients)
       
        # initialize Adam variables
        uninit_vars = []
        sess = tf.get_default_session()
        if sess is None:
            sess = tf.Session()
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.variables_initializer(uninit_vars))

    def optimize(self, input_vals_list, steps=None):
        if steps is None:
            steps = self._optimizer_steps
        sess = tf.get_default_session()
        feed_dict = dict(list(zip(self._inputs, input_vals_list)))
       
        adam_loss = sess.run(self._loss, feed_dict=feed_dict)
        logger.log("imitation_loss %s" % adam_loss)
        min_loss = adam_loss
        for i in range(steps):

            _, adam_loss = sess.run([self._train_step , self._loss], feed_dict=feed_dict)
            if i%50 == 0:
               logger.log("imitation_loss %s" % adam_loss)
        logger.record_tabular("ILLoss", adam_loss)
        return adam_loss


    def reset_optimizer(self):
        sess=tf.get_default_session()
        sess.run(self._optimizer_vars_initializers)

