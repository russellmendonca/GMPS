import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
from sandbox.rocky.tf.core.utils import make_input, make_dense_layer, forward_dense_layer, make_param_layer, \
    forward_param_layer
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian  # This is just a util class. No params.
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.spaces.box import Box
from tensorflow.python.framework import dtypes
from sandbox.rocky.tf.policies.temp_tf_future import spatial_softmax

from sandbox.rocky.tf.core.mil_utils import *

load_params = True



class MAMLGaussianMLPPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            max_std=1000.0,
            std_modifier=1.0,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.identity,
            mean_network=None,
            std_network=None,
            std_parametrization='exp',
            init_flr=1.0,
            stop_grad=False,
            extra_input_dim=0,
            im_x_dim = 48,
            im_y_dim = 64,
            latent_dim = 2,
            norm_type = None,
            policyType = 'conv_biasAda_bias'
            # metalearn_baseline=False,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :param grad_step_size: the step size taken in the learner's gradient update, sample uniformly if it is a range e.g. [0.1,1]
        :param stop_grad: whether or not to stop the gradient through the gradient.
        :return:
        """
        Serializable.quick_init(self, locals())
        #assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        self.im_x_dim , self.im_y_dim = im_x_dim , im_y_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.bias_transform_dim = latent_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, obs_dim + extra_input_dim,)
        self.extra_input_dim = extra_input_dim
        self.stop_grad = stop_grad
        self.policyType = policyType
        self.init_flr = init_flr

        self.norm_type = norm_type
        # self.metalearn_baseline = metalearn_baseline

        # create network
        self.n_conv_layers = 3 ; self.n_fc_hidden_layers = 2 

        if mean_network is None:
            self.all_params = self.create_MLP(  # TODO: this should not be a method of the policy! --> helper
                name="mean_network",
                filter_sizes = [5,3,3] , num_filters = [16,16,16] , fc_hidden_sizes = [100, 100] , output_dim = self.action_dim, num_input_channels = 3
            )
            self.strides = [(3,3) , (3,3) , (1,1)]

            self.input_tensor, _ = self.forward_MLP('mean_network', self.all_params,
                reuse=None # Need to run this for batch norm
            )
            forward_mean = lambda x, params, is_train: self.forward_MLP('mean_network', all_params=params,
                input_tensor=x, is_training=is_train)[1]
        else:
            raise NotImplementedError('Not supported.')

        if std_network is not None:
            raise NotImplementedError('Not supported.')
        else:
            if adaptive_std:
                raise NotImplementedError('Not supported.')
            else:
                if std_parametrization == 'exp':
                    init_std_param = np.log(init_std)
                elif std_parametrization == 'softplus':
                    init_std_param = np.log(np.exp(init_std) - 1)
                else:
                    raise NotImplementedError
                self.all_params['std_param'] = make_param_layer(
                    num_units=self.action_dim,
                    param=tf.constant_initializer(init_std_param),
                    name="output_std_param",
                    trainable=learn_std,
                )
                forward_std = lambda x, params: forward_param_layer(x, params['std_param'])
            self.all_param_vals = None

            # unify forward mean and forward std into a single function
            self._forward = lambda obs, params, is_train: (
                    forward_mean(obs, params, is_train), forward_std(obs, params))

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
                max_std_param = np.log(max_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
                max_std_param = np.log(np.exp(max_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param  # TODO: change these to min_std_param_raw
            self.max_std_param = max_std_param
            self.std_modifier = np.float64(std_modifier)
            #print("initializing max_std debug4", self.min_std_param, self.max_std_param)


            self._dist = DiagonalGaussian(self.action_dim)

            self._cached_params = {}

            super(MAMLGaussianMLPPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(self.input_tensor, dict(), is_training=False)
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            # pre-update policy
            self._init_f_dist = tensor_utils.compile_function(
                inputs=[self.input_tensor],
                outputs=[mean_var, log_std_var],
            )
            self._cur_f_dist = self._init_f_dist


    @property
    def vectorized(self):
        return True

    def set_init_surr_obj(self, input_list, surr_objs_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor


    def recompute_dist_for_adjusted_std(self):
        dist_info_sym = self.dist_info_sym(self.input_tensor, dict(), is_training=False)
        mean_var = dist_info_sym["mean"]
        log_std_var = dist_info_sym["log_std"]

        self._cur_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=[mean_var, log_std_var],
        )

    def compute_updated_dists(self, samples):
        """ Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
        With MAML_IL, this is only done during the testing iterations
        """
        start = time.time()
        num_tasks = len(samples)
        param_keys = self.all_params.keys()
        update_param_keys , no_update_param_keys = self.get_update_param_keys(param_keys)

    
        sess = tf.get_default_session()

        theta0_dist_info_list = []
        for i in range(num_tasks):
            if 'agent_infos_orig' not in samples[i].keys():
                assert False, "agent_infos_orig is missing--this should have been handled by process_samples"
            else:
                agent_infos_orig = samples[i]['agent_infos_orig']
            theta0_dist_info_list += [agent_infos_orig[k] for k in agent_infos_orig.keys()]


        theta_l_dist_info_list = []
        for i in range(num_tasks):
            agent_infos = samples[i]['agent_infos']
            theta_l_dist_info_list += [agent_infos[k] for k in agent_infos.keys()]

        obs_list, action_list, adv_list = [], [], []

        for i in range(num_tasks):
            if True: #not self.metalearn_baseline:
                inputs = ext.extract(samples[i],
                                     'observations', 'actions', 'advantages')
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
            

        #inputs = obs_list + action_list + adv_list
        # else:
        inputs = theta0_dist_info_list + theta_l_dist_info_list + obs_list + action_list + adv_list

        # To do a second update, replace self.all_params below with the params that were used to collect the policy.
        init_param_values = None
        if self.all_param_vals is not None:
            init_param_values = self.get_variable_values(self.all_params)

        
        for i in range(num_tasks):
            if self.all_param_vals is not None:
                self.assign_params(self.all_params, self.all_param_vals[i])

        step_sizes_sym = {}
        for key in update_param_keys:
            step_sizes_sym[key] = self.all_params[key + '_stepsize']

        if 'all_fast_params_tensor' not in dir(self):
            # make computation graph once
            self.all_fast_params_tensor = []
            for i in range(num_tasks):
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i], [self.all_params[key] for key in update_param_keys])))
                fast_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] - step_sizes_sym[key]*gradients[key] for key in update_param_keys]))
                for k in no_update_param_keys:
                    fast_params_tensor[k] = self.all_params[k]
                self.all_fast_params_tensor.append(fast_params_tensor)

        # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step
        self.all_param_vals = sess.run(self.all_fast_params_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))

        # print("debug58", type(self.all_param_vals))


        if init_param_values is not None:
            self.assign_params(self.all_params, init_param_values)

        outputs = []
        inputs = tf.split(self.input_tensor, num_tasks, 0)
        for i in range(num_tasks):
            # TODO - use a placeholder to feed in the params, so that we don't have to recompile every time.
            task_inp = inputs[i]
            info, _ = self.dist_info_sym(obs_var=task_inp, state_info_vars=dict(), all_params=self.all_param_vals[i],
                    is_training=False)

            outputs.append([info['mean'], info['log_std']])

        self._cur_f_dist = tensor_utils.compile_function(
            inputs=[self.input_tensor],
            outputs=outputs,
        )
        total_time = time.time() - start
        #logger.record_tabular("ComputeUpdatedDistTime", total_time)

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

        feed_dict = {self.assign_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)


    def switch_to_init_dist(self):
        # switch cur policy distribution to pre-update policy
        self._cur_f_dist = self._init_f_dist
        self.all_param_vals = None


    def get_update_param_keys(self, keys):

        update_param_keys = [] ; no_update_param_keys = []

        if self.policyType == 'conv_biasAda_bias':
            for key in keys:
                if 'bias' in key and 'stepsize' not in key:
                    update_param_keys.append(key)
                else:
                    no_update_param_keys.append(key)

        elif self.policyType == 'conv_fcBiasAda':
            for key in keys:
                if 'conv' not in key and 'stepsize' not in key and 'std' not in key:
                    update_param_keys.append(key)
                else:
                    no_update_param_keys.append(key)

        elif self.policyType == 'conv_no_update':
            update_param_keys = [] ; no_update_param_keys = keys

        else:
            raise AssertionError('NotImplementedError')

        return update_param_keys , no_update_param_keys

    def dist_info_sym(self, obs_var, state_info_vars=None, all_params=None, is_training=True):
        # This function constructs the tf graph, only called during beginning of meta-training
        # obs_var - observation tensor
        # mean_var - tensor for policy mean
        # std_param_var - tensor for policy std before output
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params

        mean_var, std_param_var = self._forward(obs_var, all_params, is_training)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.max_std_param is not None:
            std_param_var = tf.minimum(std_param_var, self.max_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var + np.log(self.std_modifier)
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var))) + np.log(self.std_modifier)
        else:
            raise NotImplementedError

        #print("debug3", log_std_var, self.max_std_param)
        if return_params:
            return dict(mean=mean_var, log_std=log_std_var), all_params
        else:
            return dict(mean=mean_var, log_std=log_std_var)

   

    def updated_dist_info_sym(self, task_id, surr_obj, new_obs_var, params_dict=None, is_training=True):
        """ symbolically create MAML graph, for the meta-optimization, only called at the beginning of meta-training.
        Called more than once if you want to do more than one inner grad step.
        """
   
        old_params_dict = params_dict

        if old_params_dict == None:
            old_params_dict = self.all_params
        
        param_keys = self.all_params.keys()
        update_param_keys , no_update_param_keys = self.get_update_param_keys(param_keys)

        grads = tf.gradients(surr_obj, [old_params_dict[key] for key in update_param_keys])
        if self.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [old_params_dict[key] - self.all_params[key + '_stepsize']*gradients[key] for key in update_param_keys]))
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]

        return self.dist_info_sym(new_obs_var, all_params=params_dict, is_training=is_training)



    @overrides
    def get_action(self, observation, idx=None):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        # print("debug, shape of observation", np.shape(observation))
        if self.extra_input_dim > 0:
            observation = np.concatenate((observation, np.zeros(np.shape(observation)[:-1]+(self.extra_input_dim,))), axis=-1)
        flat_obs = self.observation_space.flatten(observation)
        f_dist = self._cur_f_dist
        mean, log_std = [x[0] for x in f_dist([flat_obs])]
        rnd = np.random.normal(size=np.shape(mean))
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)


    @overrides
    def get_action_single_env(self, observation, idx=0, num_tasks=40):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        flat_obs = self.observation_space.flatten(observation)
        f_dist = self._cur_f_dist
        output = f_dist([flat_obs for _ in range(num_tasks)])
        mean, log_std = output[idx]
        rnd = np.random.normal(size=np.shape(mean))
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):   #TODO: make this work with the robot
        # this function takes a numpy array observations and outputs sampled actions.
        # Assumes that there is one observation per post-update policy distr
        flat_obs = self.observation_space.flatten_n(observations)

        result = self._cur_f_dist(flat_obs)

        if len(result) == 2:
            # NOTE - this code assumes that there aren't 2 meta tasks in a batch
            means, log_stds = result
        else:
            means = np.array([res[0] for res in result])[:,0,:]
            log_stds = np.array([res[1] for res in result])[:,0,:]
        rnd = np.random.normal(size=np.shape(means))
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)  #TODO: obtain_samples needs to receive the observations from this as well

    @property
    def distribution(self):
        return self._dist

    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables()
        else:
            params = tf.global_variables()
        #print("debug, all params", params)
        # RK: this is hacky, use when unpickling
        # params = [p for p in params if p.name.startswith('mean_network') or p.name.startswith('output_std_param')]
        
        params = [p for p in params if p.name.startswith('mean_network')] # or p.name.startswith('output_std_param')]
        params = [p for p in params if 'Adam' not in p.name]
        params = [p for p in params if 'main_optimizer' not in p.name]
        params = [p for p in params if 'temperature' not in p.name]

      
        #print("debug, perams internal", params)
        return params


    # This makes all of the parameters.
    def create_MLP(self, name, filter_sizes , num_filters , fc_hidden_sizes , output_dim, num_input_channels = 3):
    
        all_params = OrderedDict()

        assert len(filter_sizes) == len(num_filters)
        fc_hidden_sizes.append(output_dim)
        # n_conv_layers = len(filter_sizes) ; fc_hidden_sizes.append(output_dim) ; n_fc_layers = len(fc_hidden_sizes)
        
        fan_in = num_input_channels
        fc_in_shape = int(num_filters[-1]*2) + self.bias_transform_dim + self.action_dim  # Given that spatial softmax is used after the conv blocks
        
        with tf.variable_scope(name):
        
            for i in range(self.n_conv_layers): 
                all_params['conv_w%d' % (i+1)] = init_conv_weights_xavier([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='conv_w%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                all_params['conv_b%d' % (i+1)] = init_bias([num_filters[i]], name='conv_b%d' % (i+1))

                # all_params['conv_w%d' % (i+1) + '_stepsize'] = tf.Variable(self.init_flr*tf.ones_like( all_params['conv_w%d' % (i+1)]), name='conv_w%d' % (i+1)+'_stepsize')
                # all_params['conv_b%d' % (i+1) + '_stepsize'] = tf.Variable(self.init_flr*tf.ones_like( all_params['conv_b%d' % (i+1)]), name='conv_b%d' % (i+1)+'_stepsize')
                fan_in = num_filters[i]

            for i in range(self.n_fc_hidden_layers+1):
                all_params['fc_w%d' % i] = init_weights([fc_in_shape, fc_hidden_sizes[i]], name='fc_w%d' % i)
                all_params['fc_b%d' % i] = init_bias([fc_hidden_sizes[i]], name='fc_b%d' % i)

                all_params['fc_w%d' % i + '_stepsize'] = tf.Variable(self.init_flr*tf.ones_like( all_params['fc_w%d' % i]), name='fc_w%d' % i+'_stepsize')
                all_params['fc_b%d' % i + '_stepsize'] = tf.Variable(self.init_flr*tf.ones_like( all_params['fc_b%d' % i]), name='fc_b%d' % i+'_stepsize')
                
                fc_in_shape = fc_hidden_sizes[i]

            all_params['bias_transform'] = tf.Variable(tf.ones(self.bias_transform_dim), name="bias_transform")
            all_params['bias_transform_stepsize'] = tf.Variable(self.init_flr*tf.ones_like(all_params['bias_transform']), name="bias_transform_stepsize")


        return all_params

    def forward_MLP(self, name, all_params, input_tensor=None,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name , reuse=tf.AUTO_REUSE):
            if input_tensor is None:
                l_in = make_input(shape=self.input_shape, input_var=None, name='input')
            else:
                l_in = input_tensor
            
            image_input = tf.reshape(l_in[:, : 3*(self.im_x_dim*self.im_y_dim)] , (-1 , self.im_x_dim ,self.im_y_dim , 3)) ; robot_config = l_in[:, 3*(self.im_x_dim*self.im_y_dim):]
            conc_bias = tf.tile(all_params['bias_transform'][None, :], (tf.shape(l_in)[0],1))

            conv_layer = image_input

            for i in range(self.n_conv_layers):
              
                conv_layer = norm(conv2d(img=conv_layer, w=all_params['conv_w%d' % (i+1)], b=all_params['conv_b%d' % (i+1)], strides=(1,)+self.strides[i]+(1,)), \
                                norm_type=self.norm_type, is_training = is_training , activation_fn=self.hidden_nonlinearity)

            #feature_points = tf.contrib.layers.spatial_softmax( conv_layer )
            feature_points = spatial_softmax(conv_layer)
            feature_points =  tf.concat([feature_points , robot_config , conc_bias] , axis = 1)

            fc_layer = feature_points
           
            for idx in range(self.n_fc_hidden_layers):
                fc_layer = forward_dense_layer(fc_layer, all_params['fc_w'+str(idx)], all_params['fc_b'+str(idx)],
                                            batch_norm=(self.norm_type == 'batch_norm'),
                                            nonlinearity=self.hidden_nonlinearity,
                                            reuse=tf.AUTO_REUSE,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(fc_layer, all_params['fc_w'+str(self.n_fc_hidden_layers)], all_params['fc_b'+str(self.n_fc_hidden_layers)],
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

    def log_diagnostics(self, paths, prefix=''):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular(prefix+'AveragePolicyStd', np.mean(np.exp(log_stds)))

    #### code largely not used after here except when resuming/loading a policy. ####
    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        # Not used
        import pdb; pdb.set_trace()
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def get_param_dtypes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def set_param_values(self, flattened_params, all_params=False, **tags):
        debug = tags.pop("debug", False)
        # print("debug, all params", all_params) True
        # print("debug, param shapes", self.get_param_shapes(all_params, **tags))
        #TODO: remove this hacky code
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(all_params , **tags))

      
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(all_params, **tags),
                self.get_param_dtypes(all_params, **tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, all_params=False, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(all_params, **tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            # print("debug, using load params")
            d["params"] = self.get_param_values(all_params=True)
        #print("debug, shape of d params", np.shape(d['params']))
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.variables_initializer(self.get_params(all_params=True)))
            #print("debug, setstate using", d, np.shape(d['params']))

            ###############TODO : change this when temperature is removed#####################
            self.set_param_values(d["params"][:], all_params=True)
            # self.set_param_values(d["params"][:12607], all_params=True)


