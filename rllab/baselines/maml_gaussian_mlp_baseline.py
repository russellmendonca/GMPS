import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
# from sandbox.rocky.tf.regressors.maml_gaussian_mlp_regressor import MAMLGaussianMLPRegressor
# from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
# from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
# from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian  # This is just a util class. No params.

from collections import OrderedDict
from sandbox.rocky.tf.misc import tensor_utils
from tensorflow.contrib.layers.python import layers as tf_layers
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors

from sandbox.rocky.tf.core.utils import make_input, make_dense_layer, forward_dense_layer, make_param_layer, \
    forward_param_layer

import tensorflow as tf

class MAMLGaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            learning_rate=0.01,
            algo_discount=0.99,
            repeat=30,
            repeat_sym=30,
            momentum=0.5,
            hidden_sizes=(32,32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            init_meta_constant=0.0,
            normalize_inputs=True,
            normalize_outputs=True,
            extra_input_dim=0,

    ):
        Serializable.quick_init(self, locals())

        self.env_spec = env_spec
        obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, 2*(obs_dim+extra_input_dim)+3,)
        self.input_to_discard = extra_input_dim  #multiply by 0 the last extra_input_dim elements of obs vector
        self.obs_mask = np.array([1.0]*obs_dim+[0.]*extra_input_dim)
        self.learning_rate = learning_rate
        self.algo_discount = algo_discount
        self.max_path_length = 100
        self._normalize_inputs = normalize_inputs
        self._normalize_outputs = normalize_outputs


        #
        # self._enh_obs_mean_var = tf.Variable(
        #     tf.zeros((1,) + self.input_shape, dtype=tf.float32),
        #     name="enh_obs_mean",
        #     trainable=False
        # )
        # self._enh_obs_std_var = tf.Variable(
        #     tf.ones((1,) + self.input_shape, dtype=tf.float32),
        #     name="enh_obs_std",
        #     trainable=False
        # )
        self.output_dim=1
        self._ret_mean_var = tf.Variable(
            tf.zeros((self.output_dim), dtype=tf.float32),
            name="ret_mean",
            trainable=False
        )
        self._ret_std_var = tf.Variable(
            tf.ones((self.output_dim), dtype=tf.float32),
            name="ret_std",
            trainable=False
        )



        self.all_params = self.create_MLP(
            name="mean_baseline_network",
            output_dim=1,
            hidden_sizes=hidden_sizes,
        )
        self.input_tensor, _ = self.forward_MLP('mean_baseline_network', self.all_params, reuse=None)
        print("debug, input_tensor", self.input_tensor )
        self.normalized_input_tensor = normalize_sym(self.input_tensor)

        self.all_params['meta_constant'] = make_param_layer(
            num_units=1,
            param=tf.constant_initializer(init_meta_constant),
            name="output_bas_meta_constant",
            trainable=True,
        )
        forward_mean = lambda x, params, is_train: self.forward_MLP('mean_baseline_network',all_params=params, input_tensor=x, is_training=is_train)[1]
        forward_meta_constant = lambda x, params: forward_param_layer(x, params['meta_constant'])
        self._forward = lambda normalized_enh_obs, params, is_train: (forward_mean(normalized_enh_obs, params, is_train), forward_meta_constant(normalized_enh_obs, params))
        self.all_param_vals = None

        # sess = tf.get_default_session()
        # if sess is None:
        #     sess = tf.Session()
        # sess.run(tf.global_variables_initializer())

        self.learning_rate_per_param = OrderedDict(zip(self.all_params.keys(),[tf.Variable(self.learning_rate * tf.ones(tf.shape(self.all_params[key])), trainable=False) for key in self.all_params.keys()]))
        # sess.run(tf.global_variables_initializer())
        self.accumulation = OrderedDict(zip(self.all_params.keys(),[tf.Variable(tf.zeros(tf.shape(self.all_params[key])), trainable=False) for key in self.all_params.keys()]))
        # self.last_grad = OrderedDict(zip(self.all_params.keys(),[tf.Variable(tf.zeros_like(self.all_params[key]), trainable=False) for key in self.all_params.keys()]))


        # self._dist = DiagonalGaussian(1)
        self._cached_params = {}
        super(MAMLGaussianMLPBaseline, self).__init__(env_spec)

        normalized_predict_sym = self.normalized_predict_sym(normalized_enh_obs_vars=self.normalized_input_tensor)
        mean_var = normalized_predict_sym['mean'] * self._ret_std_var + self._ret_mean_var
        meta_constant_var = normalized_predict_sym['meta_constant']

        self._init_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=[mean_var,meta_constant_var],
        )
        self._cur_f_dist = self._init_f_dist
        self.initialized = 30
        self.lr_mult = 1.0
        self.repeat=repeat
        self.repeat_sym=repeat_sym
        self.momentum = momentum

        # self.momopt = tf.train.MomentumOptimizer(learning_rate=0.000001, momentum=0.999)
        # self.momopt = tf.train.AdamOptimizer(name="bas_optimizer")


        # sess = tf.get_default_session()
        # if sess is None:
        #     sess = tf.Session()
        # keys = self.all_params.keys()
        # sess.run(tf.variables_initializer([self.all_params[k] for k in keys]))
        # sess.run(tf.variables_initializer([self.learning_rate_per_param[k] for k in keys]))
        # sess.run(tf.variables_initializer([self.accumulation[k] for k in keys]))
        # sess.run(tf.global_variables_initializer())
        # uninit_vars = []
        # sess = tf.get_default_session()
        # if sess is None:
        #     sess = tf.Session()
        # for var in tf.global_variables():
        #     # note - this is hacky, may be better way to do this in newer TF.
        #     try:
        #         sess.run(var)
        #     except tf.errors.FailedPreconditionError:
        #         uninit_vars.append(var)
        # sess.run(tf.variables_initializer(uninit_vars))



    @property
    def vectorized(self):
        return True


    def set_init_surr_obj(self, input_list, surr_obj_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        if surr_obj_tensor is not None:
            self.surr_obj = surr_obj_tensor
        else:
            assert len(input_list) == 2
            enh_obs, returns_vars = input_list[0], input_list[1]
            normalized_enh_obs = normalize_sym(enh_obs)
            normalized_returns = normalize_sym(returns_vars, debug=True)
            normalized_predicted_returns_sym, _ = self.normalized_predict_sym(normalized_enh_obs_vars=normalized_enh_obs,all_params=self.all_params)
            predicted_returns_means_sym = tf.reshape(normalized_predicted_returns_sym['mean'], [-1])
            meta_constant_sym = tf.reshape(normalized_predicted_returns_sym['meta_constant'], [-1])
            baseline_pred_loss = tf.reduce_mean(tf.square(predicted_returns_means_sym - normalized_returns)) - 0.0 * tf.reduce_mean(meta_constant_sym)
            self.surr_obj = baseline_pred_loss


    @overrides
    def fit(self, paths, log=True, repeat=None):  # TODO REVERT repeat=10000
        # return True
        repeat = repeat if repeat is not None else self.repeat
        if 'surr_obj' not in dir(self):
            assert False, "why didn't we define it already"
        if self.initialized > 0:
            repeat = 400
            self.lr_mult = 0.5
        """Equivalent of compute_updated_dists"""
        update_param_keys = self.all_params.keys()
        no_update_param_keys = []

        sess = tf.get_default_session()

        if 'init_params_tensor' not in dir(self):
            self.init_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] for key in update_param_keys]))
        self.init_param_vals = sess.run(self.init_params_tensor)
        self.init_accumulation_vals = sess.run(self.accumulation)
        # self.init_grad_vals = sess.run(self.last_grad)
        obs = np.concatenate([np.clip(p["observations"],-10,10) for p in paths])
        obs = np.multiply(obs,self.obs_mask)
        obs2 = np.concatenate([np.square(np.clip(p["observations"],-10,10)) for p in paths])
        obs2 = np.multiply(obs2, self.obs_mask)
        al = np.concatenate([np.arange(len(p["rewards"])).reshape(-1, 1)/100.0 for p in paths])
        al2 =al**2
        al3 = al**3
        enh_obs = np.concatenate([obs,obs2,al,al2,al3],axis=1)
        returns = np.concatenate([p["returns"] for p in paths])  #TODO: do we need to reshape the returns here?
        inputs = [enh_obs] + [returns]

        # if self._normalize_inputs:
        #     sess.run([
        #         tf.assign(self._enh_obs_mean_var, np.mean(enh_obs, axis=0, keepdims=True)),
        #         tf.assign(self._enh_obs_std_var, np.std(enh_obs, axis=0, keepdims=True) + 1e-8),
        #     ])

        if self._normalize_outputs:
            sess.run([
                tf.assign(self._ret_mean_var, np.mean(returns, axis=0, keepdims=True)),
                tf.assign(self._ret_std_var, np.std(returns, axis=0, keepdims=True) + 1e-8),
            ])

        # enh_obs_mean = np.mean(enh_obs, axis=0, keepdims=True)
        # enh_obs_std = np.std(enh_obs, axis=0, keepdims=True) + 1e-8
        # normalized_enh_obs = (enh_obs-enh_obs_mean)/enh_obs_std

        if 'all_fast_params_tensor' not in dir(self) or self.all_fast_params_tensor is None:
            gradients = OrderedDict(zip(update_param_keys, tf.gradients(self.surr_obj, [self.all_params[key] for key in update_param_keys])))
            new_accumulation = {key:gradients[key] + self.momentum * self.accumulation[key]  for key in update_param_keys}
            # new_accumulation = {key:gradients[key] + (self.last_grad[key]+gradients[key])**4/(self.last_grad[key]**4+gradients[key]**4+1e-8) * self.momentum * self.accumulation[key] for key in update_param_keys}
            # new_accumulation = {key:gradients[key] + tf.divide(self.accumulation[key]*(gradients[key]**2) - gradients[key]**3,tf.square(self.accumulation[key]-gradients[key])+1e-4) for key in update_param_keys}
            # new_accumulation = {key:self.momentum * self.accumulation[key] +gradients[key] * tf.divide((self.accumulation[key] + gradients[key])**2,self.accumulation[key]**2+gradients[key]**2) for key in update_param_keys}
            # new_accumulation = {key:gradients[key] + self.accumulation[key] * tf.divide((2 * gradients[key] - self.accumulation[key]) * (self.accumulation[key] - gradients[key]),self.accumulation[key]**2 + gradients[key]**2+1e-8) for key in update_param_keys}
            fast_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] - self.lr_mult * self.learning_rate_per_param[key]*new_accumulation[key] for key in update_param_keys]))
            # new_accumulation = {key:gradients[key] for key in update_param_keys}
            for k in no_update_param_keys:
                fast_params_tensor[k] = self.all_params[k]
            self.all_fast_params_tensor = (fast_params_tensor, new_accumulation)
            # pull new param vals out of tensorflow, so gradient computation only done once
            # these are the updated values of the params after the gradient step
        for _ in range(repeat):
            self.all_param_vals, self.accumulation_vals= sess.run(self.all_fast_params_tensor,
                                           feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
            self.assign_params(self.all_params, self.all_param_vals)
            self.assign_accumulation(self.accumulation, self.accumulation_vals)
            # self.assign_gradients(self.last_grad, new_gradient_vals)

        # if init_param_values is not None:
        #     self.assign_params(self.all_params, init_param_values)

        normalized_input_vars = tf.split(self.normalized_input_tensor, 1, 0)  #TODO: how to convert this since we don't need to calculate multiple updates simultaneously
        normalized_enh_obs_vars = normalized_input_vars[0]
        normalized_predict_sym, _ = self.normalized_predict_sym(normalized_enh_obs_vars=normalized_enh_obs_vars, all_params=self.all_param_vals,is_training=False)

        outputs = [normalized_predict_sym['mean']*self._ret_std_var + self._ret_mean_var, normalized_predict_sym['meta_constant']]

        self._cur_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=outputs,
        )
        if self.initialized > 0:
            self.init_param_vals = sess.run(self.init_params_tensor)
            self.initialized -= 1
            if self.initialized == 0:
                self.all_fast_params_tensor = None
                self.lr_mult = 1.0

        self.assign_accumulation(self.accumulation, self.init_accumulation_vals)

    def get_variable_values(self, tensor_dict):
        sess = tf.get_default_session()
        result = sess.run(tensor_dict)
        return result

    def assign_params(self, tensor_dict, param_values):
        if 'assign_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_placeholders = {}
            self.assign_ops = {}
            for key in tensor_dict.keys():
                self.assign_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_ops[key] = tf.assign(tensor_dict[key], self.assign_placeholders[key])

        # print("debug78,", tensor_dict.keys())
        # print("debug79,", tensor_dict)
        # print("debug80,", param_values)

        feed_dict = {self.assign_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)

    def assign_lr(self, tensor_dict, param_values):
        if 'assign_lr_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_lr_placeholders = {}
            self.assign_lr_ops = {}
            for key in tensor_dict.keys():
                self.assign_lr_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_lr_ops[key] = tf.assign(tensor_dict[key], self.assign_lr_placeholders[key])

        feed_dict = {self.assign_lr_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_lr_ops, feed_dict)

    def assign_accumulation(self, tensor_dict, param_values):
        if 'assign_acc_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_acc_placeholders = {}
            self.assign_acc_ops = {}
            for key in tensor_dict.keys():
                self.assign_acc_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_acc_ops[key] = tf.assign(tensor_dict[key], self.assign_acc_placeholders[key])

        feed_dict = {self.assign_acc_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_acc_ops, feed_dict)



    def assign_gradients(self, tensor_dict, param_values):
        if 'assign_grad_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_grad_placeholders = {}
            self.assign_grad_ops = {}
            for key in tensor_dict.keys():
                self.assign_grad_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_grad_ops[key] = tf.assign(tensor_dict[key], self.assign_grad_placeholders[key])

        feed_dict = {self.assign_grad_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_grad_ops, feed_dict)


    @overrides
    def predict(self, path):
        # flat_obs = self.env_spec.observation_space.flatten_n(path['observations'])
        obs = np.clip(path['observations'],-10,10)
        obs = np.multiply(obs,self.obs_mask)
        obs2 = np.square(obs) # no need to do the mask here
        # al = np.zeros(shape=(len(path["rewards"]),1))
        al = np.arange(len(path["rewards"])).reshape(-1, 1)/100.0
        al2 = al**2
        al3 = al**3
        # al0 = al**0

        enh_obs = np.concatenate([obs, obs2, al, al2, al3],axis=1)
        # enh_obs = np.concatenate([al, al2, al3],axis=1)
        # print("debug24", enh_obs)
        # print("debug24.1", np.shape(enh_obs))
        result = self._cur_f_dist(enh_obs)
        if len(result) == 2:
            means, meta_constant = result
        else:
            raise NotImplementedError('Not supported.')
        return np.reshape(means, [-1])

    def meta_predict(self, observations):
        # flat_obs = self.env_spec.observation_space.flatten_n(path['observations'])
        obs = np.zeros(shape=np.shape(observations))
        obs2 = obs
        # al = np.zeros(shape=(len(path["rewards"]),1))
        al = np.zeros(shape=(len(observations),1))
        al2 = al
        al3 = al
        # al0 = al

        enh_obs = np.concatenate([obs, obs2, al, al2, al3],axis=1)
        # enh_obs = np.concatenate([al, al2, al3],axis=1)
        # print("debug24", enh_obs)
        # print("debug24.1", np.shape(enh_obs))
        result = self._cur_f_dist(enh_obs)
        if len(result) == 2:
            means, meta_constant = result # meta constant is actually a repetition list of the same constant
        else:
            raise NotImplementedError('Not supported.')
        return np.reshape(meta_constant, [-1])


    # @property
    # def distribution(self):
    #     return self._dist

    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables()
        else:
            params = tf.global_variables()

        params = [p for p in params if p.name.startswith('mean_baseline_network') or p.name.startswith('output_bas_meta_constant')]
        params = [p for p in params if 'Adam' not in p.name]
        params = [p for p in params if 'bas_optimizer' not in p.name]

        return params


        # This makes all of the parameters.
    def create_MLP(self, name, output_dim, hidden_sizes,
                   hidden_W_init=tf_layers.xavier_initializer(), hidden_b_init=tf.zeros_initializer(),
                   output_W_init=tf_layers.xavier_initializer(), output_b_init=tf.zeros_initializer(),
                   weight_normalization=False,
                   ):
        all_params = OrderedDict()

        cur_shape = self.input_shape
        with tf.variable_scope(name):
            for idx, hidden_size in enumerate(hidden_sizes):
                W, b, cur_shape = make_dense_layer(
                    cur_shape,
                    num_units=hidden_size,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_norm=weight_normalization,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
            W, b, _ = make_dense_layer(
                cur_shape,
                num_units=output_dim,
                name='output',
                W=output_W_init,
                b=output_b_init,
                weight_norm=weight_normalization,
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b' + str(len(hidden_sizes))] = b

        return all_params

    def forward_MLP(self, name, all_params, input_tensor=None,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                l_in = make_input(shape=self.input_shape, input_var=None, name='input')
            else:
                l_in = input_tensor

            l_hid = l_in

            for idx in range(self.n_hidden):
                l_hid = forward_dense_layer(l_hid, all_params['W' + str(idx)], all_params['b' + str(idx)],
                                            batch_norm=batch_normalization,
                                            nonlinearity=self.hidden_nonlinearity,
                                            scope=str(idx), reuse=reuse,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid, all_params['W' + str(self.n_hidden)],
                                         all_params['b' + str(self.n_hidden)],
                                         batch_norm=False, nonlinearity=self.output_nonlinearity,
                                         )
            return l_in, output



    def get_params(self, all_params=False, **tags):
        """
        Get the list of parameters (symbolically), filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(all_params, **tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self, all_params=False, **tags):
        params = self.get_params(all_params, **tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def switch_to_init_dist(self):
        # switch cur baseline distribution to pre-update baseline
        self._cur_f_dist = self._init_f_dist
        self.all_param_vals = None
        self.assign_params(self.all_params,self.init_param_vals)
        self.assign_accumulation(self.accumulation, self.init_accumulation_vals)
        # self.assign_gradients(self.last_grad, self.init_grad_vals)

    # def predict_sym(self, enh_obs_vars, all_params=None, is_training=True):
    #     """equivalent of dist_info_sym, this function constructs the tf graph, only called
    #     during beginning of meta-training"""
    #     return_params = True
    #     if all_params is None:
    #         return_params = False
    #         all_params = self.all_params
    #         if self.all_params is None:
    #             assert False, "Shouldn't get here"
    #
    #
    #     mean_var, meta_constant_var = self._forward(normalized_enh_obs=enh_obs_vars, params=all_params, is_train=is_training)
    #
    #     if return_params:
    #         return dict(mean=mean_var, meta_constant=meta_constant_var), all_params
    #     else:
    #         return dict(mean=mean_var, meta_constant=meta_constant_var)


    def normalized_predict_sym(self, normalized_enh_obs_vars, all_params=None, is_training=True):
        """equivalent of dist_info_sym, this function constructs the tf graph, only called
        during beginning of meta-training"""
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params
            if self.all_params is None:
                assert False, "Shouldn't get here"


        normalized_mean_var, normalized_meta_constant_var = self._forward(normalized_enh_obs=normalized_enh_obs_vars, params=all_params, is_train=is_training)

        if return_params:
            return dict(mean=normalized_mean_var, meta_constant=normalized_meta_constant_var), all_params
        else:
            return dict(mean=normalized_mean_var, meta_constant=normalized_meta_constant_var)



    def updated_n_predict_sym(self, baseline_pred_loss, n_enh_obs_vars, params_dict=None, accumulation_sym=None):
        """ symbolically create post-fitting baseline predict_sym, to be used for meta-optimization.
        Equivalent of updated_dist_info_sym"""
        old_params_dict = params_dict
        start_params_dict = old_params_dict
        if old_params_dict is None:
            old_params_dict = self.all_params
        param_keys = self.all_params.keys()

        update_param_keys = param_keys
        no_update_param_keys = []
        grads = tf.gradients(baseline_pred_loss, [old_params_dict[key] for key in update_param_keys])
        gradients = dict(zip(update_param_keys, grads))
        if accumulation_sym is not None:
            new_accumulation_sym = {key:self.momentum * accumulation_sym[key] + gradients[key] for key in update_param_keys}
            params_dict = OrderedDict(zip(update_param_keys, [old_params_dict[key] - self.lr_mult * self.learning_rate_per_param[key] * new_accumulation_sym[key] for key in update_param_keys]))
        else:
            new_accumulation_sym = None
            params_dict = OrderedDict(zip(update_param_keys, [old_params_dict[key] - self.lr_mult * self.learning_rate_per_param[key] * gradients[key] for key in update_param_keys]))
        # for key in update_param_keys:
        #     old_params_dict[key] = params_dict[key]
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]
        return self.normalized_predict_sym(normalized_enh_obs_vars=n_enh_obs_vars, all_params=params_dict), new_accumulation_sym

    def build_adv_sym(self,enh_obs_vars,rewards_vars, returns_vars, all_params, baseline_pred_loss=None, repeat=None):  # path_lengths_vars was before all_params
        assert baseline_pred_loss is None, "don't give me baseline pred loss"
        repeat = repeat if repeat is not None else self.repeat_sym
        updated_params = all_params
        normalized_enh_obs_vars = normalize_sym(enh_obs_vars)
        ret_mean_var_sym, ret_var_var_sym = tf.nn.moments(returns_vars, axes=[0])
        ret_std_var_sym = tf.sqrt(ret_var_var_sym)
        normalized_returns_vars = (returns_vars - ret_mean_var_sym)/(ret_std_var_sym + 1e-8)
        normalized_returns_vars_ = tf.reshape(normalized_returns_vars, [-1,1])
        accumulation_sym = self.accumulation
        i = tf.constant(0)
        keys = updated_params.keys()
        updated_params_list = [updated_params[key] for key in keys ]
        accumulation_sym_list = [accumulation_sym[key] for key in keys]
        while_loop_vars_0 = [updated_params_list, accumulation_sym_list, i]
        c = lambda _, ___, i: i < repeat

        def get_structure(x):
            if "get_shape" in dir(x):
                return x.get_shape()
            else:
                if isinstance(x, OrderedDict):
                    return OrderedDict({key:get_structure(x[key]) for key in x.keys()})
                elif isinstance(x, dict):
                    return {key:get_structure(x[key]) for key in x.keys()}
                elif isinstance(x, list):
                    return [get_structure(y) for y in x]


        def b(updated_params_list, accumulation_sym_list, i ):
            updated_params_reconstr = OrderedDict(zip(keys,updated_params_list))
            accumulation_sym_reconstr = OrderedDict(zip(keys, accumulation_sym_list))
            n_predicted_returns_sym, _ = self.normalized_predict_sym(normalized_enh_obs_vars=normalized_enh_obs_vars, all_params=updated_params_reconstr)
            baseline_pred_loss = tf.reduce_mean(tf.square(n_predicted_returns_sym['mean'] - normalized_returns_vars_) + 0.0 * n_predicted_returns_sym['meta_constant'])
            (n_predicted_returns_sym, updated_params1), accumulation_sym1 = self.updated_n_predict_sym(baseline_pred_loss=baseline_pred_loss, n_enh_obs_vars=normalized_enh_obs_vars, params_dict=updated_params_reconstr, accumulation_sym=accumulation_sym_reconstr)  # TODO: do we need to update the params here?
            updated_params_list = [updated_params1[key] for key in keys]
            accumulation_sym_list = [accumulation_sym1[key] for key in keys]
            return [updated_params_list, accumulation_sym_list, i+1]
        # print("debug",[get_structure(x) for x in while_loop_vars_0])
        shape_invariants = [get_structure(x) for x in while_loop_vars_0]
        (updated_params_list, accumulation_sym_list, i) = tf.while_loop(c,b,while_loop_vars_0, shape_invariants=shape_invariants)
        updated_params = OrderedDict(zip(keys, updated_params_list))
        accumulation_sym = OrderedDict(zip(keys, accumulation_sym_list))
        n_predicted_returns_sym, _ = self.normalized_predict_sym(normalized_enh_obs_vars=normalized_enh_obs_vars, all_params=updated_params)
        # predicted_returns_sym = n_predicted_returns_sym * self._ret_std_var + self._ret_mean_var
        organized_rewards = tf.reshape(rewards_vars, [-1,self.max_path_length])
        organized_pred_returns = tf.reshape(n_predicted_returns_sym['mean'] *(ret_std_var_sym+1e-8) + ret_mean_var_sym , [-1,self.max_path_length])
        organized_pred_returns_ = tf.concat((organized_pred_returns[:,1:], tf.reshape(tf.zeros(tf.shape(organized_pred_returns[:,0])),[-1,1])),axis=1)

        deltas = organized_rewards + self.algo_discount * organized_pred_returns_ - organized_pred_returns
        adv_vars = tf.map_fn(lambda x: discount_cumsum_sym(x, self.algo_discount), deltas)

        adv_vars = tf.reshape(adv_vars, [-1])
        adv_vars = (adv_vars - tf.reduce_mean(adv_vars))/tf.sqrt(tf.reduce_mean(adv_vars**2))  # centering advantages
        adv_vars = adv_vars + n_predicted_returns_sym['meta_constant'][0]

        return adv_vars

    @overrides
    def set_param_values(self, flattened_params, **tags):
        raise NotImplementedError("todo")

        # @overrides
        # def fit(self, paths, log=True):  # aka compute updated baseline
        #     # self._preupdate_params = self._regressor.get_param_values()
        #
        #     param_keys = self.all_params.keys()
        #     update_param_keys = param_keys
        #     no_update_param_keys = []
        #     sess = tf.get_default_session()
        #
        #     observations = np.concatenate([p["observations"] for p in paths])
        #     returns = np.concatenate([p["returns"] for p in paths])
        #
        #     inputs = observations + returns
        #
        #
        #     learning_rate = self.learning_rate
        #     if self.all_param_vals is not None:
        #         self.assign_params(self.all_params, self.all_param_vals)
        #
        #     if "fit_tensor" not in dir(self):
        #         gradients = dict(zip(update_param_keys, tf.gradients(self._regressor.loss_sym, [self.all_params[key] for key in update_param_keys])))
        #         self.fit_tensor = OrderedDict(zip(update_param_keys,
        #                                              [self.all_params[key] - learning_rate * gradients[key] for key in
        #                                               update_param_keys]))
        #         for k in no_update_param_keys:
        #             self.fit_tensor[k] = self.all_params[k]
        #
        #     self.all_param_vals = sess.run(self.fit_tensor, feed_dict = dict(list(zip(self.input_list_for_grad, inputs))))
        #
        #
        #     inputs = self.input_tensor
        #     task_inp = inputs
        #     output = self.predict_sym(task_inp, dict(),all_params=self.all_param_vals, is_training=False)
        #
        #
        #     self._regressor._f_predict = tensor_utils.compile_function(inputs=[self.input_tensor], outputs=output)


        #
    # def revert(self):
    #     # assert self._preupdate_params is not None, "already reverted"
    #     if self._preupdate_params is None:
    #         return
    #     else:
    #         self._regressor.set_param_values(self._preupdate_params)
    #         self._preupdate_params = None

    # def compute_updated_baseline(self, samples):
    #     """ Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
    #     """
    #     num_tasks = len(samples)
    #     param_keys = self.all_params.keys()
    #     update_param_keys = param_keys
    #     no_update_param_keys = []
    #
    #     sess = tf.get_default_session()
    #
    #
    #
    #     for i in range(num_tasks):
    #
    #
    #     self._cur_f_dist = tensor_utils.compile_function



'''
    def fit_train_baseline(self, paths, repeat=100):
        if 'surr_obj' not in dir(self):
            assert False, "why didn't we define it already"
        param_keys = self.all_params.keys()

        sess = tf.get_default_session()
        obs = np.concatenate([np.clip(p["observations"],-10,10) for p in paths])
        obs2 = np.concatenate([np.square(np.clip(p["observations"],-10,10)) for p in paths])
        al = np.concatenate([np.arange(len(p["rewards"])).reshape(-1, 1)/100.0 for p in paths])
        al2 =al**2
        al3 = al**3
        # al0 = al**0
        returns = np.concatenate([p["returns"] for p in paths])
        # inputs = [np.concatenate([al,al2,al3],axis=1)] + [returns]
        inputs = [np.concatenate([obs,obs2,al,al2,al3],axis=1)] + [returns]

        if 'lr_train_step' not in dir(self) :
            gradients = dict(zip(param_keys, tf.gradients(self.surr_obj, [self.all_params[key] for key in param_keys])))  #+[self.learning_rate_per_param[key] for key in self.learning_rate_per_param.keys()])))
            postupdate_params = OrderedDict(zip(param_keys, [self.all_params[key] - self.learning_rate_per_param[key]*gradients[key] for key in param_keys]))
            print("debug88\n", self.all_params)
            print("debug89\n", postupdate_params)
            predicted_returns_sym, _ = self.predict_sym(enh_obs_vars = self.input_list_for_grad[0],all_params=postupdate_params)
            print("debug01\n", self.input_list_for_grad[0])
            print("debug02\n", self.input_list_for_grad[1])
            loss_after = tf.reduce_mean(tf.square(predicted_returns_sym['mean'] - tf.reshape(self.input_list_for_grad[1], [-1,1])) + 0.0 * predicted_returns_sym['meta_constant'])
            self.lr_train_step = self.momopt.minimize(loss=loss_after, var_list=[self.learning_rate_per_param[key] for key in self.learning_rate_per_param.keys()])
            # self.lr_train_step = self.momopt.minimize(loss=loss_after) #, var_list=[self.learning_rate_per_param[key] for key in self.learning_rate_per_param.keys()])
                                            # OrderedDict(zip(param_keys, [self.all_params[key] - self.learning_rate_per_param[key] * gradients[key] for key in param_keys]))
            # pull new param vals out of tensorflow, so gradient computation only done once
            # these are the updated values of the params after the gradient step

        uninit_vars = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.variables_initializer(uninit_vars))
        feed_dict = dict(list(zip(self.input_list_for_grad, inputs)))

        for _ in range(repeat):
            if _ in [0,repeat-1]:
                print("debug99", sess.run(self.learning_rate_per_param).items())
            sess.run(self.lr_train_step,feed_dict=feed_dict)

            # self.all_param_vals, self.learning_rate_per_param_vals = sess.run(self.all_fast_params_tensor2,
            #                                 feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
            # self.assign_params(self.all_params, self.all_param_vals)
            # self.assign_lr(self.learning_rate_per_param, self.learning_rate_per_param_vals)
            #
        #
        # enh_obs = self.input_list_for_grad[0]
        # info, _ = self.predict_sym(enh_obs_vars=enh_obs, is_training=False)
        #
        # outputs = [info['mean'], info['meta_constant']]
        #
        # self._cur_f_dist = tensor_utils.compile_function(
        #     inputs=[self.input_tensor],
        #     outputs=outputs,
        # )
'''


def discount_cumsum_sym(var, discount):
    # y[0] = x[0] + discount * x[1] + discount**2 * x[2] + ...
    # y[1] = x[1] + discount * x[2] + discount**2 * x[3] + ...
    discount = tf.cast(discount, tf.float32)
    range_ = tf.cast(tf.range(tf.size(var)), tf.float32)
    var_ = var * tf.pow(discount, range_)
    return tf.cumsum(var_,reverse=True) * tf.pow(discount,-range_)


def normalize_sym(x, debug=False):
    mean, var = tf.nn.moments(x, axes=[0])
    if debug:
        print("debug, normalize_sym", mean, var)
    return (x - mean) / (tf.sqrt(var) + 1e-8)



