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
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
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

        if self.policy.use_prob_latents:
            self.latent_dist = DiagonalGaussian(self.policy.latent_dim)



    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        obs_vars, action_vars, expert_action_vars, adv_vars,  noise_vars, task_family_idx_vars = [], [], [], [], [],[] 
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))

            expert_action_vars.append(self.env.action_space.new_tensor_variable(
                name='expert_actions' + stepnum + '_' + str(i),
                extra_dims=1,
            ))

            adv_vars.append(tensor_utils.new_tensor(
                name='advantage' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
            noise_vars.append(tf.placeholder(dtype=tf.float32, shape=[None, self.policy.latent_dim], name='noise' + stepnum + '_' + str(i)))
            task_family_idx_vars.append(tensor_utils.new_tensor(
                name='task_family_idx' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.int32,
            ))
        return obs_vars, action_vars, expert_action_vars , adv_vars, noise_vars, task_family_idx_vars


    def make_vars_latent(self, stepnum='0'):
        # lists over the meta_batch_size
        adv_vars, z_vars, task_family_idx_vars = [], [], []
        for i in range(self.meta_batch_size):
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage_latent' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
            z_vars.append(tf.placeholder(dtype=tf.float32, shape=[None, self.policy.latent_dim], name='zs_latent' + stepnum + '_' + str(i)))
            task_family_idx_vars.append(tensor_utils.new_tensor(
                name='task_family_idx_latents' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.int32,
            ))
        return adv_vars, z_vars, task_family_idx_vars


    @overrides
    def init_opt(self):
        assert not int(self.policy.recurrent)  # not supported
        assert self.use_maml  # only maml supported

        dist = self.policy.distribution

        theta0_dist_info_vars, theta0_dist_info_vars_list = [], []
        if self.policy.use_prob_latents:
            for i in range(self.meta_batch_size):
            
                theta0_dist_info_vars.append({'mean' : tf.placeholder(tf.float32, shape=[None] + [self.policy.latent_dim], name='theta0_'+str(i)+'_mean'), \
                                       'log_std': tf.placeholder(tf.float32, shape=[None] + [self.policy.latent_dim], name='theta0_'+str(i)+'_log_std') })
                theta0_dist_info_vars_list += [theta0_dist_info_vars[i][k] for k in ['mean' , 'log_std']]

        else:
            for i in range(self.meta_batch_size):
                theta0_dist_info_vars.append({
                    k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='theta0_%s_%s' % (i, k))
                    for k, shape in dist.dist_info_specs
                    })
                theta0_dist_info_vars_list += [theta0_dist_info_vars[i][k] for k in dist.dist_info_keys]

        all_surr_objs, all_surr_objs_slow, input_vars_list, inner_input_vars_list = [], [], [], []
        new_params = []
        input_vars_list += tuple(theta0_dist_info_vars_list)
        #inner_input_vars_list += tuple(theta0_dist_info_vars_list) + tuple(theta_l_dist_info_vars_list)

        for grad_step in range(self.num_grad_updates):  # we are doing this for all but the last step
            
            inner_surr_objs, inner_surr_objs_simple , new_params = [], [] , [] 

            if self.policy.use_prob_latents:
        
                adv_vars_latent, z_vars_latent, task_family_idx_vars_latent = self.make_vars_latent(str(grad_step))    
                for i in range(self.meta_batch_size): 

                    dist_info_vars_latent = {"mean": tf.gather(self.policy.all_params['latent_means'], task_family_idx_vars_latent[i]), \
                                            "log_std": tf.gather(self.policy.all_params['latent_stds'], task_family_idx_vars_latent[i])}

                           
                    new_params.append(self.policy.all_params)
                   
                    logli_latent = self.latent_dist.log_likelihood_sym(z_vars_latent[i], dist_info_vars_latent)

                    theta_circle = OrderedDict({key: tf.stop_gradient(self.policy.all_params[key]) for key in ['latent_means' , 'latent_stds']})

                    dist_info_vars_latent_circle = {"mean": tf.gather(theta_circle['latent_means'], task_family_idx_vars_latent[i]), \
                                        "log_std": tf.gather(theta_circle['latent_stds'], task_family_idx_vars_latent[i])}

                    lr_per_step_fast = dist.likelihood_ratio_sym(z_vars_latent[i], theta0_dist_info_vars[i], dist_info_vars_latent_circle)
                    lr_per_step_fast = self.ism(lr_per_step_fast)

                    # formulate a minimization problem
                    # The gradient of the surrogate objective is the policy gradient
                    inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli_latent, lr_per_step_fast), adv_vars_latent[i])))
                    inner_surr_objs_simple.append(-tf.reduce_mean(tf.multiply(logli_latent, adv_vars_latent[i])))

                inner_input_vars_list += adv_vars_latent + z_vars_latent + task_family_idx_vars_latent
                input_vars_list       += adv_vars_latent + z_vars_latent + task_family_idx_vars_latent

            else:
                obs_vars, action_vars, expert_action_vars , adv_vars, _, task_family_idx_vars = self.make_vars(str(grad_step))
                for i in range(self.meta_batch_size): 

                    noise_vars = tf.tile( tf.zeros((1, self.policy.latent_dim)) ,  (tf.shape(obs_vars[i])[0], 1) )       
                    dist_info_vars, params = self.policy.dist_info_sym(obs_vars[i], task_family_idx_vars[i], noise_vars, state_info_vars={}, all_params=self.policy.all_params)        
                    new_params.append(params)

                    logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)
                    keys = self.policy.all_params.keys()
                    theta_circle = OrderedDict({key: tf.stop_gradient(self.policy.all_params[key]) for key in keys})
                    dist_info_vars_circle, _ = self.policy.dist_info_sym(obs_vars[i], task_family_idx_vars[i], noise_vars, state_info_vars = {}, all_params=theta_circle)
                    
                    lr_per_step_fast = dist.likelihood_ratio_sym(action_vars[i], theta0_dist_info_vars[i], dist_info_vars_circle)
                    import ipdb
                    ipdb.set_trace()
                    lr_per_step_fast = self.ism(lr_per_step_fast)

                    # formulate a minimization problem
                    # The gradient of the surrogate objective is the policy gradient
                    inner_surr_objs.append(-tf.reduce_mean(tf.multiply(tf.multiply(logli, lr_per_step_fast), adv_vars[i])))
                    inner_surr_objs_simple.append(-tf.reduce_mean(tf.multiply(logli, adv_vars[i])))

                inner_input_vars_list += obs_vars + action_vars + adv_vars +  task_family_idx_vars 
                input_vars_list       += obs_vars + action_vars + adv_vars +  task_family_idx_vars 


            # For computing the fast update for sampling
            # At this point, inner_input_vars_list is theta0 + theta_l + obs + action + adv
            self.policy.set_init_surr_obj(inner_input_vars_list, inner_surr_objs_simple)

            all_surr_objs.append(inner_surr_objs)
        
        

        ###################################Outer loop imitation step ##############################################################################
        obs_vars, action_vars, expert_action_vars , _, noise_vars, task_family_idx_vars = self.make_vars('test')
        outer_surr_objs = []  ; self.updated_latent_means = [] ; self.updated_latent_stds = []
        for i in range(self.meta_batch_size):  
            
            dist_info_sym_i, updated_params = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], obs_vars[i], task_family_idx_vars[i], noise_vars[i], params_dict=new_params[i])
            
            self.updated_latent_means.append(updated_params['latent_means'])
            self.updated_latent_stds.append(updated_params['latent_stds'])

            a_star = expert_action_vars[i]
            s = dist_info_sym_i["log_std"]
            m = dist_info_sym_i["mean"]
            outer_surr_obj = tf.reduce_mean(m**2 - 2*m*a_star+a_star**2+self.l2loss_std_multiplier*(tf.square(tf.exp(s))))
            outer_surr_objs.append(outer_surr_obj)

        #################################################################################################################################################
        
        outer_surr_obj = tf.reduce_mean(tf.stack(outer_surr_objs, 0)) # mean over all the different tasks
        input_vars_list += obs_vars + action_vars + expert_action_vars + noise_vars + task_family_idx_vars


        target = [self.policy.all_params[key] for key in self.policy.all_params.keys() if key not in ['latent_means', 'latent_stds']]
        #target = [self.policy.all_params[key] for key in self.policy.all_params.keys()]
       
        self.optimizer.update_opt(
            loss=outer_surr_obj,
            target=target,
            inputs=input_vars_list,
        )

        return dict()


#######################################
    @overrides
    def optimize_policy(self, itr, all_samples_data):
        assert itr not in TESTING_ITRS
        assert len(all_samples_data) >= self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!
        assert self.use_maml
        sess = tf.get_default_session()
        input_vals_list = []

        # Code to account for off-policy sampling when more than 1 beta steps
        theta0_dist_info_list = []
        if self.policy.use_prob_latents:
            dist_info_keys = ['mean', 'log_std'] 
        else:
            dist_info_keys = self.policy.distribution.dist_info_keys


        for i in range(self.meta_batch_size):
            if 'dist_infos_orig' not in all_samples_data[0][i].keys():
                assert False, "dist_infos_orig is missing--this should have been handled in batch_maml_polopt"
            else:
                dist_infos_orig = all_samples_data[0][i]['dist_infos_orig']
            theta0_dist_info_list += [dist_infos_orig[k] for k in dist_info_keys]

        input_vals_list += tuple(theta0_dist_info_list)
        
        ############################Inner Loop Data for RL step ####################################
        for step in range(self.num_grad_updates):  
            obs_list, action_list, adv_list, task_family_idx_list        = [], [], [], []
            adv_list_latent , z_list_latent, task_family_idx_list_latent = [], [], []
          
            if self.policy.use_prob_latents:

                for i in range(self.meta_batch_size):

                    inputs = ext.extract(
                        all_samples_data[step][i],
                        "advantages", "noises", "task_family_idxs"
                    )
                    means = tf.gather(self.policy.all_params['latent_means'], inputs[-1])
                    stds = tf.gather(self.policy.all_params['latent_stds'], inputs[-1])
                    zs = sess.run(means + inputs[-2]*tf.exp(stds))
                    adv_list_latent.append(inputs[0])
                    z_list_latent.append(zs)
                    task_family_idx_list_latent.append(inputs[2])

                input_vals_list += adv_list_latent + z_list_latent + task_family_idx_list_latent
                
            else:
                for i in range(self.meta_batch_size):
                    inputs = ext.extract(
                        all_samples_data[step][i],
                        "observations", "actions", "advantages",  "task_family_idxs"
                    )
                    obs_list.append(inputs[0])
                    action_list.append(inputs[1])
                    adv_list.append(inputs[2])
                    task_family_idx_list.append(inputs[3])

                input_vals_list += obs_list + action_list + adv_list + task_family_idx_list 
        ############################################################################################

        ###################Outer loop data for imitation ###################################
       
        obs_list, action_list, expert_action_list , noise_list , task_family_idx_list = [], [], [] , [], [] 
        for i in range(self.meta_batch_size):  # for each task
            inputs = ext.extract(
                all_samples_data[self.num_grad_updates][i],
                "observations", "actions", "expert_actions", "noises",  "task_family_idxs"
            )
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            expert_action_list.append(inputs[2])
            noise_list.append(inputs[3])
            task_family_idx_list.append(inputs[4])

        input_vals_list += obs_list + action_list + expert_action_list + noise_list + task_family_idx_list
        #########################################################################################
        
        feed_dict = dict(list(zip(self.optimizer._inputs, input_vals_list)))
        updated_latent_means = sess.run(self.updated_latent_means , feed_dict = feed_dict)
        updated_latent_stds  = np.exp(sess.run(self.updated_latent_stds  , feed_dict = feed_dict))
        print('############# Means ####################')
        for i in updated_latent_means: print(i)
        print('############# Stds #####################')
        for i in updated_latent_stds: print(i)

     
        steps = self.adam_curve[min(itr,len(self.adam_curve)-1)]
        logger.log("Optimizing using %s Adam steps on itr %s" % (steps, itr))
        start_loss = self.optimizer.optimize(input_vals_list, steps=steps )
        return start_loss
       
    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        debug_params = self.policy.get_params_internal()

        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )






