from collections import OrderedDict

from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np

class VPG(BatchPolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """
    def __init__(
            self,
            env,
            policy,
            baseline,
            default_step,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.default_step_size = default_step
        self.opt_info = None
        super(VPG, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)
   

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            name='advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]
        self.input_list_for_grad = [obs_var, action_var, advantage_var] + state_info_vars_list

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - tf.reduce_sum(logli * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            max_kl = tf.reduce_max(kl * valid_var)
        else:
            surr_obj = - tf.reduce_mean(logli * advantage_var)
            mean_kl = tf.reduce_mean(kl)
            max_kl = tf.reduce_max(kl)
        
        self.surr_obj = surr_obj

    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
      
        
        
        self.optimize(inputs, tf.get_default_session(),  itr)
       

    @overrides
    def optimize(self, inputs, sess, itr):
        param_keys = []
        step_keys=[]
        all_keys = list(self.policy.all_params.keys())
       
        for key in all_keys:
            if ('stepsize' not in key):
                param_keys.append(key)
            else:
                step_keys.append(key)

        update_param_keys = param_keys


        no_update_param_keys = []

        
        step_sizes_sym = {}
        for key in param_keys:
           
            step_sizes_sym[key] = self.policy.all_params[key + '_stepsize']

       
        gradients = dict(zip(update_param_keys, tf.gradients(self.surr_obj, [self.policy.all_params[key] for key in update_param_keys])))
        update_tensor = OrderedDict(zip(update_param_keys, [self.policy.all_params[key] - step_sizes_sym[key]*gradients[key] for key in update_param_keys]))
       

        # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step
        result = sess.run(update_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))

    
        opList = []
        for key in param_keys:     
            opList.append(self.policy.all_params[key].assign(result[key]))
           
           
    
        #if (itr>=1) and (itr<=5):
        # if itr >= 1:
        #     for key in step_keys:
        #         stepSize = sess.run(self.policy.all_params[key])
        #         #stepSize = np.zeros(shape = stepSize.shape)
        #         opList.append(self.policy.all_params[key].assign(stepSize/2))
       
        for assign_op in opList:
            sess.run(assign_op)
       
       
    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
