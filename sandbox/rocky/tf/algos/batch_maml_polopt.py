from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.sampler.stateful_pool import singleton_pool
from copy import deepcopy
import matplotlib
matplotlib.use('Pdf')
import itertools


import matplotlib.pyplot as plt
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time
import numpy as np
import random as rd
import joblib
from rllab.misc.tensor_utils import split_tensor_dict_list, stack_tensor_dict_list
# from maml_examples.reacher_env import fingertip
from rllab.sampler.utils import rollout
from maml_examples.maml_experiment_vars import TESTING_ITRS, PLOT_ITRS, VIDEO_ITRS, BASELINE_TRAINING_ITRS
#from maml_examples import pusher_env

class BatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            metalearn_baseline=False,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=100,
            max_path_length=500,
            meta_batch_size=100,
            num_grad_updates=1,
            num_grad_updates_for_testing=1,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            make_video=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            use_maml=True,
            use_maml_il=False,
            test_on_training_goals=False,
            limit_demos_num=None,
            test_goals_mult=1,
            load_policy=None,
            pre_std_modifier=1.0,
            post_std_modifier=1.0,
            
            goals_to_load=None,
            goals_pool_to_load=None,
            expert_trajs_dir=None,
            expert_trajs_suffix="",
            goals_pickle_to=None,
            goals_pool_size=None,
            use_pooled_goals=True,
            extra_input=None,
            extra_input_dim=0,
            seed=1,
            debug_pusher=False,
            comet_logger=None,
            outer_iteration=0,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  #
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.outer_iteration=outer_iteration
        self.comet_logger = comet_logger
        self.seed=seed
        self.env = env
        self.policy = policy

        self.load_policy = load_policy
        self.baseline = baseline
        self.metalearn_baseline = metalearn_baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size is the number of trajectories for one fast grad update.
        # self.batch_size is the number of total transitions to collect.
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        # self.beta_steps = beta_steps
        # self.beta_curve = beta_curve if beta_curve is not None else [self.beta_steps]
        self.old_il_loss = None
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.make_video = make_video
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size  # number of tasks
        self.num_grad_updates = num_grad_updates  # number of gradient steps during training
        self.num_grad_updates_for_testing = num_grad_updates_for_testing  # number of gradient steps during training
        self.use_maml_il = use_maml_il
        self.test_on_training_goals= test_on_training_goals
        self.testing_itrs = TESTING_ITRS
        logger.log("test_on_training_goals %s" % self.test_on_training_goals)
        self.limit_demos_num = limit_demos_num
        self.test_goals_mult = test_goals_mult
        self.pre_std_modifier = pre_std_modifier
        self.post_std_modifier = post_std_modifier
        
        #   self.action_limiter_multiplier = action_limiter_multiplier
        self.expert_trajs_dir = expert_trajs_dir
        self.expert_trajs_suffix = expert_trajs_suffix
        self.use_pooled_goals = use_pooled_goals
        self.extra_input = extra_input
        self.extra_input_dim = extra_input_dim
        self.debug_pusher=debug_pusher
        self.cached_demos=None
        self.cached_demos_path=None
        # Next, we will set up the goals and potentially trajectories that we plan to use.
        # If we use trajectorie
        self.num_tasks = self.meta_batch_size
        self.contexts = None

        self.goals_idxs_for_itr_dict = {}
        for i in range(self.n_itr):
            self.goals_idxs_for_itr_dict[i] = np.arange(0 , self.meta_batch_size)

        self.demos_path = expert_trajs_dir

        # inspecting goals_idxs_for_itr_dict
        assert set(range(self.start_itr, self.n_itr)).issubset(set(self.goals_idxs_for_itr_dict.keys())), \
            "Not all meta-iteration numbers have idx_dict in %s" % goals_pool_to_load
        for itr in range(self.start_itr, self.n_itr):
            num_goals = len(self.goals_idxs_for_itr_dict[itr])
            assert num_goals >= self.meta_batch_size, "iteration %s contained %s goals when at least %s are needed" % (itr, num_goals, self.meta_batch_size)
            self.goals_idxs_for_itr_dict[itr] = self.goals_idxs_for_itr_dict[itr][:self.meta_batch_size]

        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BatchSampler
                print("Using Batch Sampler")
            else:
                sampler_cls = VectorizedSampler
                print("Using Vectorized Sampler")
        if sampler_args is None:
            sampler_args = dict()
        if 'n_envs' not in sampler_args.keys():
            sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)


    def start_worker(self):
        self.sampler.start_worker()
      
    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix='',testitr=False, preupdate=False, contexts = None):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)

        paths = self.sampler.obtain_samples(itr=itr, reset_args=reset_args, return_dict=True, log_prefix=log_prefix, \
                                preupdate=preupdate , contexts = contexts)
        assert type(paths) == dict
        return paths

    def load_expert_traces(self):
       
        self.expert_traces = {taskidx : joblib.load(self.expert_trajs_dir+str(taskidx)+".pkl") for taskidx in range(self.num_tasks)}
        for taskidx in range(self.num_tasks):
            for path in self.expert_traces[taskidx]:
                if 'expert_actions' not in path.keys() :
                    path['expert_actions'] = np.clip(deepcopy(path['actions']), -1.0, 1.0)

                path['agent_infos'] = dict(mean=[[0.0] * len(path['actions'][0])]*len(path['actions']),log_std=[[0.0] * len(path['actions'][0])]*len(path['actions']))


    def process_samples(self, itr, paths, prefix='', log=True, fast_process=False, testitr=False, metalearn_baseline=False):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log, fast_process=fast_process, testitr=testitr, metalearn_baseline=metalearn_baseline, comet_logger=self.comet_logger)

    def train(self):
        # TODO - make this a util

        flatten_list = lambda l: [item for sublist in l for item in sublist]
        config = tf.ConfigProto()
      
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
        # with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
            #tf.set_random_seed(1)
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                loaded_data = joblib.load(self.load_policy)
                self.policy = loaded_data['policy']
                self.baseline = loaded_data['baseline']
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            # sess.run(tf.global_variables_initializer())
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)

            sess.run(tf.variables_initializer(uninit_vars))
           
            self.start_worker()
           
            start_time = time.time()
            self.metaitr=0
            self.load_expert_traces()
            
            for itr in range(self.start_itr, self.n_itr):
                if self.comet_logger:
                    # self.comet_logger.set_step(itr + (self.outer_iteration * self.n_itr)) ### Later we should add something to this
                    self.comet_logger.set_step(itr + self.outer_iteration)
                itr_start_time = time.time()
                np.random.seed(self.seed+itr)
                tf.set_random_seed(self.seed+itr)
                rd.seed(self.seed+itr)
                with logger.prefix('itr #%d | ' % itr):
                    all_paths_for_plotting = []
                    all_postupdate_paths = []
                    num_inner_updates = self.num_grad_updates_for_testing if itr in self.testing_itrs else self.num_grad_updates
                   

                    expert_traj_for_metaitr =  {newIdx  : self.expert_traces[oldIdx][:self.limit_demos_num] for newIdx , oldIdx in enumerate(self.goals_idxs_for_itr_dict[itr])}
                 
                    self.policy.std_modifier = self.pre_std_modifier
                    self.policy.switch_to_init_dist()  # Switch to pre-update policy
                    #import ipdb; ipdb.set_trace()
                    #print(sess.run(self.policy.all_params['bias_transformation']))

                    
                    if itr in self.testing_itrs:
                        env = self.env
                        while 'sample_goals' not in dir(env):
                            env = env.wrapped_env

                        goal_idxs_to_use = self.goals_idxs_for_itr_dict[itr]
                    
                    all_samples_data = []
                    for step in range(num_inner_updates+1): # inner loop
                        logger.log("Obtaining samples...")


                        if step < num_inner_updates:
                            
                            paths = self.obtain_samples(itr=itr, reset_args=goal_idxs_to_use,
                                                            log_prefix=str(step),testitr=itr in self.testing_itrs,preupdate=True)
                            paths = store_agent_infos(paths)  # agent_infos_orig is populated here

                        elif itr in self.testing_itrs:
                            
                            paths = self.obtain_samples(itr=itr, reset_args=goal_idxs_to_use,
                                                                log_prefix=str(step),testitr=True,preupdate=False , contexts = self.contexts)

                        else:
                            paths = expert_traj_for_metaitr

                        logger.log("Processing samples...")
                        samples_data = {}
                        for tasknum in paths.keys():  # the keys are the tasks
                            # don't log because this will spam the console with every task.
                            if self.use_maml_il and step == num_inner_updates:
                                fast_process = True
                            else:
                                fast_process = False
                            if itr in self.testing_itrs:
                                testitr = True
                            else:
                                testitr = False
                            samples_data[tasknum] = self.process_samples(itr, paths[tasknum], log=False, fast_process=fast_process, testitr=testitr, metalearn_baseline=self.metalearn_baseline)

                        all_samples_data.append(samples_data)
                        # for logging purposes only
                        self.process_samples(itr, flatten_list(paths.values()), prefix=str(step), log=True, fast_process=True, testitr=testitr, metalearn_baseline=self.metalearn_baseline)
                        if step == num_inner_updates:
                            logger.record_tabular("AverageReturnLastTest", self.sampler.memory["AverageReturnLastTest"],front=True)  #TODO: add functionality for multiple grad steps
                            logger.record_tabular("TestItr", ("1" if testitr else "0"),front=True)
                            logger.record_tabular("MetaItr", self.metaitr,front=True)
                        if self.comet_logger:
                            self.comet_logger.log_metric("AverageReturnLastTest",
                                                  self.sampler.memory["AverageReturnLastTest"])
                            # self.comet_logger.log_metric("TestItr", ("1" if testitr else "0"))
                            self.comet_logger.log_metric("MetaItr", self.metaitr)
                        # logger.log("Logging diagnostics...")
                        # self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))

                        for taskIdx in range(len(paths)):
                            prefix =  'stage'+str(step)+'_'+'task'+str(taskIdx)
                            self.log_diagnostics( paths[taskIdx] , prefix = prefix)

                        if step == num_inner_updates-1:
                            if itr not in self.testing_itrs:
                               
                                self.policy.std_modifier = self.post_std_modifier*self.policy.std_modifier
                            else:
                               
                                self.policy.std_modifier = self.post_std_modifier*self.policy.std_modifier
                            if (itr in self.testing_itrs or not self.use_maml_il or step<num_inner_updates-1) and step < num_inner_updates:
                                # do not update on last grad step, and do not update on second to last step when training MAMLIL
                                logger.log("Computing policy updates...")
                                self.policy.compute_updated_dists(samples=samples_data)

                    logger.log("Optimizing policy...")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    start_loss = self.optimize_policy(itr, all_samples_data)

                    if itr not in self.testing_itrs:
                        self.metaitr += 1

                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data[-1])  # , **kwargs)
                    
                    if self.store_paths:
                        params["paths"] = all_samples_data[-1]["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    logger.dump_tabular(with_prefix=False)

                    # The rest is some example plotting code.
                    # Plotting code is useful for visualizing trajectories across a few different tasks.
                    
        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix, comet_logger=self.comet_logger)
        #self.policy.log_diagnostics(paths, prefix)
        #self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)


def store_agent_infos(paths):
    tasknums = paths.keys()
    for t in tasknums:
        for path in paths[t]:
            path['agent_infos_orig'] = deepcopy(path['agent_infos'])
    return paths

