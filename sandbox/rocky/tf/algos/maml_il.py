import tensorflow as tf
import numpy as np
import rllab.misc.logger as logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_maml_polopt import BatchMAMLPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from maml_examples.maml_experiment_vars import TESTING_ITRS, BASELINE_TRAINING_ITRS
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
from collections import OrderedDict


class MAMLIL(BatchMAMLPolopt):

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            use_maml=True,
            beta_steps=1,
            adam_steps=1,
            adam_curve=None,
            l2loss_std_mult=1.0,
            importance_sampling_modifier=tf.identity,
            metalearn_baseline=False,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(min_penalty=1e-8)
            optimizer = QuadDistExpertOptimizer("main_optimizer", adam_steps=adam_steps, use_momentum_optimizer=False)  #  **optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.adam_curve = adam_curve if adam_curve is not None else [adam_steps]
        self.use_maml = use_maml
        self.l2loss_std_multiplier = l2loss_std_mult
        self.ism = importance_sampling_modifier
        #self.old_start_il_loss = None

        if "extra_input" in kwargs.keys():
            self.extra_input = kwargs["extra_input"]
        else:
            self.extra_input = ""
        if "extra_input_dim" in kwargs.keys():
            self.extra_input_dim = kwargs["extra_input_dim"]
        else:
            self.extra_input_dim = 0

        super(MAMLIL, self).__init__(optimizer=optimizer, beta_steps=beta_steps, use_maml_il=True,  **kwargs)


    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        # We should only need the last stepnum for meta-optimization.
        obs_vars, action_vars, adv_vars, rewards_vars, returns_vars, path_lengths_vars, expert_action_vars = [], [], [], [], [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
                add_to_flat_dim=(0 if self.extra_input is None else self.extra_input_dim),
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                    'advantage' + stepnum + '_' + str(i),
                    ndim=1, dtype=tf.float32,
                ))
            
            expert_action_vars.append(self.env.action_space.new_tensor_variable(
                name='expert_actions' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
        
        return obs_vars, action_vars, adv_vars, expert_action_vars
       
    @overrides
    def init_opt(self):
        assert not int(self.policy.recurrent)  # not supported
        assert self.use_maml  # only maml supported

        dist = self.policy.distribution

        theta0_dist_info_vars, theta0_dist_info_vars_list = [], []
        for i in range(self.meta_batch_size):
            theta0_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='theta0_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            theta0_dist_info_vars_list += [theta0_dist_info_vars[i][k] for k in dist.dist_info_keys]

        state_info_vars, state_info_vars_list = {}, []  # TODO: is this needed?

        all_surr_objs, all_surr_objs_slow, input_vars_list, inner_input_vars_list = [], [], [], []
        new_params = []
        input_vars_list += tuple(theta0_dist_info_vars_list)
        #inner_input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)

        for grad_step in range(self.num_grad_updates):  # we are doing this for all but the last step
           
            obs_vars, action_vars, adv_vars, expert_action_vars = self.make_vars(str(grad_step))
            inner_surr_objs, inner_surr_objs_simple, inner_surr_objs_sym = [], [], []  # surrogate objectives
           
            new_params = []

            for i in range(self.meta_batch_size):  # for training task T_i
                adv = adv_vars[i]
                dist_info_sym_i, params = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=self.policy.all_params)

                new_params.append(params)
                logli_i = dist.log_likelihood_sym(action_vars[i], dist_info_sym_i)
                keys = self.policy.all_params.keys()
                theta_circle = OrderedDict({key: tf.stop_gradient(self.policy.all_params[key]) for key in keys})
                dist_info_sym_i_circle, _ = self.policy.dist_info_sym(obs_vars[i], state_info_vars, all_params=theta_circle)
                lr_per_step_fast = dist.likelihood_ratio_sym(action_vars[i], theta0_dist_info_vars[i], dist_info_sym_i_circle)
                lr_per_step_fast = self.ism(lr_per_step_fast)

                # formulate a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                
                inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_i, lr_per_step_fast), adv)))
                inner_surr_objs_simple.append(-tf.reduce_mean(tf.multiply(logli_i, adv)))
                
            inner_input_vars_list += obs_vars + action_vars + adv_vars
            input_vars_list += obs_vars + action_vars + adv_vars
            # For computing the fast update for sampling
            # At this point, inner_input_vars_list is theta0 + theta_l + obs + action + adv
            self.policy.set_init_surr_obj(inner_input_vars_list, inner_surr_objs_simple)

            input_vars_list += expert_action_vars # TODO: is this pre-update expert action vars? Should we kill this?
            all_surr_objs.append(inner_surr_objs)

        # LAST INNER GRAD STEP
        obs_vars, action_vars, _, expert_action_vars = self.make_vars('test')  # adv_vars was here instead of _
        
        outer_surr_objs = [] ; biases = []
        for i in range(self.meta_batch_size):  # here we cycle through the last grad update but for validation tasks (i is the index of a task)
            

            dist_info_sym_i, updated_params_i = self.policy.updated_dist_info_sym(task_id=i,surr_obj=all_surr_objs[-1][i],new_obs_var=obs_vars[i], params_dict=new_params[i])
            
            if self.post_policy:
                bs = tf.shape(obs_vars[i])[0]
                
                bias = updated_params_i['bias_transformation']
                biases.append(bias)
                tiled_bias = tf.tile(bias[None, :], (bs,1)) + 1e-1*tf.random_normal((bs, 2))
                #tiled_bias = tf.tile(bias[None, :], (bs,1))
                
                contextual_obs = tf.concat([obs_vars[i], tiled_bias], axis=1)
                dist_info_sym_i = self.post_policy.dist_info_sym(contextual_obs)
               
                self.biases = biases
            #updated_params.append(updated_params_i)
            # # here we define the loss for meta-gradient
            a_star = expert_action_vars[i]
            s = dist_info_sym_i["log_std"]
            m = dist_info_sym_i["mean"]
            outer_surr_obj = tf.reduce_mean(m**2 - 2*m*a_star+a_star**2+self.l2loss_std_multiplier*(tf.square(tf.exp(s))))
            outer_surr_objs.append(outer_surr_obj)

        outer_surr_obj = tf.reduce_mean(tf.stack(outer_surr_objs, 0)) # mean over all the different tasks
        input_vars_list += obs_vars + action_vars + expert_action_vars 
        target = [self.policy.all_params[key] for key in self.policy.all_params.keys()]

        if self.post_policy:
            bias_contraint = tf.reduce_mean((biases[0] - biases[2])**2 + (biases[1] - biases[3])**2)
            outer_surr_obj+=10*bias_contraint
            target = [self.policy.all_params[key] for key in self.policy.all_params.keys() if key!='bias_transformation']
            target.extend([self.post_policy.all_params[key] for key in self.post_policy.all_params.keys()])

        self.optimizer.update_opt(
            loss=outer_surr_obj,
            target=target,
            inputs=input_vars_list,
        )

        return dict()


#######################################
    @overrides
    def optimize_policy(self, itr, all_samples_data):
        assert len(all_samples_data) >= self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!
        assert self.use_maml

        input_vals_list = []

        # Code to account for off-policy sampling when more than 1 beta steps
        theta0_dist_info_list = []
        for i in range(self.meta_batch_size):
            if 'agent_infos_orig' not in all_samples_data[0][i].keys():
                assert False, "agent_infos_orig is missing--this should have been handled in batch_maml_polopt"
            else:
                agent_infos_orig = all_samples_data[0][i]['agent_infos_orig']
            theta0_dist_info_list += [agent_infos_orig[k] for k in self.policy.distribution.dist_info_keys]
        input_vals_list += tuple(theta0_dist_info_list)

        for step in range(self.num_grad_updates):
            obs_list, action_list, adv_list, rewards_list, returns_list, path_lengths_list, expert_action_list = [], [], [], [], [], [], []
            for i in range(self.meta_batch_size):  # for each task
        
                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "advantages", "expert_actions",
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
                expert_action_list.append(inputs[3])
                
            input_vals_list += obs_list + action_list + adv_list + expert_action_list
            

        for step in [self.num_grad_updates]:  # last step
            obs_list, action_list, expert_action_list = [], [], []  # last step's adv_list not currently used in maml_il
            for i in range(self.meta_batch_size):  # for each task
                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "expert_actions",
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                expert_action_list.append(inputs[2])

            input_vals_list += obs_list + action_list + expert_action_list


        logger.log("Computing loss before")
       # loss_before = self.optimizer.loss(input_vals_list)
        if itr not in TESTING_ITRS:
            steps = self.adam_curve[min(itr,len(self.adam_curve)-1)]
            logger.log("Optimizing using %s Adam steps on itr %s" % (steps, itr))
            start_loss = self.optimizer.optimize(input_vals_list, steps=steps )
            # self.optimizer.optimize(input_vals_list)
            return start_loss

        else:
            logger.log("Not Optimizing")
            logger.record_tabular("ILLoss",float('nan'))
            return None
       
    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        debug_params = self.policy.get_params_internal()

        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )






