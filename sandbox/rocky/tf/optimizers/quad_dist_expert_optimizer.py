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
    Runs Tensorflow optimization on a quadratic loss function, under the kl constraint

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
        self._max_constraint_val = None
        self._constraint_name = None
        self._optimizer_steps = adam_steps
        self._correction_term = None
        self._use_momentum_optimizer=use_momentum_optimizer


    def update_opt(self, loss, target, leq_constraint, inputs, constraint_name="constraint", dummy_loss = None, *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        constraint_term, constraint_value = leq_constraint

        # print("debug101", constraint_term)
        with tf.variable_scope(self._name):
            penalty_var = tf.placeholder(tf.float32, tuple(), name="penalty")
        # temp1 = penalty_var * constraint_term
        # temp2 = loss + penalty_var
        # temp3 = loss + constraint_term
        penalized_loss = loss + penalty_var * constraint_term

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._inputs = inputs
        self._loss = loss
        self._dummy_loss = dummy_loss

        if self._use_momentum_optimizer:
            self._adam=tf.train.MomentumOptimizer(learning_rate=0.0000001, momentum=0.997, name=self._name)
            assert False, "not supported at the moment"
        else:
            self._adam = tf.train.AdamOptimizer(name=self._name) #learning_rate=0.001
        self._optimizer_vars_initializers = [var.initializer for var in tf.global_variables() if self._name in var.name]

        # self._adam = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.5)
        # self._train_step = self._adam.minimize(self._loss)

        if "correction_term" in kwargs:
            self._correction_term = kwargs["correction_term"]
        else:
            self._correction_term = None

        self._gradients = self._adam.compute_gradients(loss=self._loss, var_list=self._target)
        # self._gradients = self._adam.compute_gradients(loss=self._loss)
        if self._correction_term is None:
            self._train_step = self._adam.apply_gradients(self._gradients)
        else:
            # print("debug1", self._gradients)
            # print("debug2", self._correction_term)
            # print("debug2", self._correction_term.keys())
            self.new_gradients = []
            for grad,var in self._gradients:
                if var in self._correction_term.keys():
                    self.new_gradients.append((grad + self._correction_term[var], var))
                else:
                    self.new_gradients.append((grad,var))
            # print("debug3", self.new_gradients)
            self._train_step = self._adam.apply_gradients(self.new_gradients)

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




        def get_opt_output():
            params = target.get_params(trainable=True)
            grads = tf.gradients(penalized_loss, params)
            for idx, (grad, param) in enumerate(zip(grads, params)):
                if grad is None:
                    grads[idx] = tf.zeros_like(param)
            flat_grad = tensor_utils.flatten_tensor_variables(grads)
            return [
                tf.cast(penalized_loss, tf.float32),
                tf.cast(flat_grad, tf.float32),
            ]

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs, loss, log_name="f_loss"),
            f_constraint=lambda: tensor_utils.compile_function(inputs, constraint_term, log_name="f_constraint"),
            f_penalized_loss=lambda: tensor_utils.compile_function(
                inputs=inputs + [penalty_var],
                outputs=[penalized_loss, loss, constraint_term],
                log_name="f_penalized_loss",
            ),
            f_opt=lambda: tensor_utils.compile_function(
                inputs=inputs + [penalty_var],
                outputs=get_opt_output(),
            )
        )

    def loss(self, inputs):
        return self._opt_fun["f_loss"](*inputs)

    def constraint_val(self, inputs):
        return self._opt_fun["f_constraint"](*inputs)

    def optimize(self, input_vals_list, steps=None):
        if steps is None:
            steps = self._optimizer_steps
        sess = tf.get_default_session()
        feed_dict = dict(list(zip(self._inputs, input_vals_list)))

        # print("debug01, tf gradients", sess.run(self._gradients, feed_dict=feed_dict)[0][0][0][0:4])
        # print("debug01, tf gradients", sess.run(self.new_gradients, feed_dict=feed_dict)[0][0][0][0:4])
        # numeric_grad = compute_numeric_grad(loss=self._loss, params=self._target.all_params, feed_dict=feed_dict)
        # print("debug02", numeric_grad)
        # print("debug543", feed_dict.keys())
        # for key in feed_dict.keys():
        #     if 'obs' in key.name:
                # print("debug567", key, np.shape(feed_dict[key]))
       
        adam_loss = sess.run(self._loss, feed_dict=feed_dict)
        logger.log("imitation_loss %s" % adam_loss)
        min_loss = adam_loss
        for i in range(steps):
            # BREAKER: limit the amount of adam steps you take by measuring the reduction in loss. Quite costly.
            # loss_val = sess.run(self._loss, feed_dict=feed_dict)
            # print("currentloss", loss_val, i)
            # # dummy_loss_val = sess.run(self._dummy_loss, feed_dict=feed_dict)
            # # print("current dummy_loss",dummy_loss_val)
            # min_loss = min(min_loss,loss_val)
            # if loss_val > min_loss + 0.0*(init_loss-min_loss):
            #     print("breaking at", i)
            #     break

            # print("debug01", sess.run(self._gradients, feed_dict=feed_dict)[0][0][0][0:4])
            # print("debug02", sess.run(self._correction_term, feed_dict=feed_dict)[0][0][0:4])
                # print("debug03", sess.run(self.new_gradients, feed_dict=feed_dict))
          
            _, adam_loss = sess.run([self._train_step , self._loss], feed_dict=feed_dict)
            if i%50 == 0:
                logger.log("imitation_loss %s" % adam_loss)
        logger.record_tabular("ILLoss", adam_loss)
        return adam_loss


    def reset_optimizer(self):
        sess=tf.get_default_session()
        sess.run(self._optimizer_vars_initializers)

def compute_numeric_grad(loss, params, feed_dict, epsilon=1e-10):
    sess = tf.get_default_session()
    # we used this code to debug the gradients
    # grad = tf.gradients(ys=loss, xs=[params[key] for key in params.keys()])
    # print("debug02, tf gradients again", sess.run(grad, feed_dict=feed_dict))
    loss_theta = sess.run(loss, feed_dict=feed_dict)
    output = OrderedDict({})
    for key in ["W0"]: # params.keys():
        shape = sess.run(tf.shape(params[key]))
        output[key] = np.zeros(shape=shape,dtype=np.float32)
        assert len(shape) < 3, "not supported"
        if len(shape) == 1:
            for i in range(len(shape)):
                sess.run(tf.assign(params[key][i], params[key][i]+epsilon))
                loss_thetaeps = sess.run(loss, feed_dict=feed_dict)
                output[key][i] = (loss_thetaeps-loss_theta)/epsilon
                sess.run(tf.assign(params[key][i], params[key][i]-epsilon))
        if len(shape) == 2:
            a,b = shape
            for i in range(1) : # range(a*b):
                j = i % a
                k = int((i-j)/a)
                for e in [epsilon]:  #[epsilon*0.0001, epsilon*0.001, epsilon*0.01, epsilon*0.1, epsilon*1.0, epsilon*10.0, epsilon*100.0]:
                    sess.run(tf.assign(params[key], params[key]+eps_j_k(e,a,b,j,k)))
                    loss_thetaeps = sess.run(loss, feed_dict=feed_dict)
                    output[key][j][k] = (loss_thetaeps-loss_theta)/e
                    if key == "W0" and j == 0 and k <= 3:
                        print("debug1 numeric grad", key, e, j, k, output[key][j][k])
                        print("loss before", loss_theta)
                        print("loss  after", loss_thetaeps)
                    # sess.run(tf.assign(params[key], params[key]-eps_j_k(e,a,b,j,k)))
                if i ==0 and k == 4:
                    break
    return output


def eps_j_k(epsilon,a,b,j,k):
    out = np.zeros(shape=(a,b),dtype=np.float64)
    out[j][k]=epsilon
    return out

def eps_i(epsilon,a,i):
    out = np.zeros(shape=(a,),dtype=np.float64)
    out[i] = epsilon
    return out