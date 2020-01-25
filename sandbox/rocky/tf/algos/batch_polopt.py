import time

import tensorflow as tf
import numpy as np
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os.path as osp
from rllab.sampler.utils import rollout


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            batch_size_expert_traj=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            load_policy=None,
            make_video=True,
            action_noise_train=0.0,
            action_noise_test=0.0,
            reset_arg=None,
            save_expert_traj_dir=None,
            save_img_obs=False,
            goals_pool_to_load=None,
            extra_input=None,
            extra_input_dim=0,
            log_dir = None,
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
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
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
        self.outer_iteration = outer_iteration
        self.comet_logger=comet_logger
        self.env = env
        self.policy = policy
        self.load_policy = load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.batch_size_train = batch_size
        self.batch_size_expert_traj = batch_size_expert_traj
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.log_dir = log_dir
        if sampler_cls is None:
            #if self.policy.vectorized and not force_batch_sampler:
            sampler_cls = VectorizedSampler
            #else:
            #sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.reset_arg = reset_arg
        self.action_noise_train = action_noise_train
        self.action_noise_test = action_noise_test
        self.make_video = make_video
        self.save_expert_traj_dir = save_expert_traj_dir
        self.save_img_obs = save_img_obs
        self.extra_input = extra_input
        self.extra_input_dim = extra_input_dim
        if goals_pool_to_load is not None:
            self.goals_pool = joblib.load(goals_pool_to_load)['goals_pool']
            self.goals_for_ET_dict = {t:[goal] for t, goal in enumerate(self.goals_pool)}
            self.expert_traj_itrs_to_pickle = list(range(len(self.goals_pool)))  # we go through
            assert save_expert_traj_dir is not None, "please provide a filename to save expert trajectories"
            assert set(self.expert_traj_itrs_to_pickle).issubset(set(range(self.start_itr,
                                                                           self.n_itr))), "Will not go through all itrs that need to be pickled, widen the start_itr and n_itr range"
            Path(self.save_expert_traj_dir).mkdir(parents=True, exist_ok=True)
            # joblib_dump_safe(dict(goals_pool=self.goals_pool, idxs_dict=self.goals_idxs_for_itr_dict), self.save_expert_traj_dir + "goals_pool.pkl") deprecated to remove redundancy of goals_pool files
        else:
            self.goals_for_ET_dict = {}
            self.expert_traj_itrs_to_pickle = []
            assert save_expert_traj_dir is None, "can't save ETs without goals provided"


    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, preupdate=True):
        if reset_args is None:
            reset_args = self.reset_arg
        # return self.sampler.obtain_samples(itr, reset_args=reset_args, save_img_obs=self.save_img_obs) # we'll just save the whole paths and keep the observations
        print("debug, obtaining samples")
        return self.sampler.obtain_samples(itr=itr, reset_args=reset_args, return_dict=False, preupdate=preupdate)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths, comet_logger=self.comet_logger)

    def train(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config = config) as sess:
            if self.load_policy is not None:
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            # initialize uninitialized vars (I know, it's ugly)
           
            uninit_vars = []
            for var in tf.all_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.initialize_variables(uninit_vars))
            #sess.run(tf.initialize_all_variables())
            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                if self.comet_logger:
                    self.comet_logger.set_step(itr + self.outer_iteration)
                if itr == self.n_itr-1:
                    self.policy.std_modifier = 0.00001
                    #self.policy.std_modifier = 1
                    self.policy.recompute_dist_for_adjusted_std()
                if itr in self.goals_for_ET_dict.keys():
                    # self.policy.std_modifier = 0.0001
                    # self.policy.recompute_dist_for_adjusted_std()
                    goals = self.goals_for_ET_dict[itr]
                    noise = self.action_noise_test
                    self.batch_size = self.batch_size_expert_traj
                else:
                    if self.reset_arg is None:
                        goals = [None]
                    else:
                        goals = [self.reset_arg]
                    noise = self.action_noise_train
                    self.batch_size = self.batch_size_train
                paths_to_save = {}
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):

                    logger.log("Obtaining samples...")
                    
                    
                    preupdate = True if itr < self.n_itr-1 else False
                        # paths_for_goal = self.obtain_samples(itr=itr, reset_args=[{'goal': goal, 'noise': noise}])  # when using oracle environments with changing noise, use this line!
                    paths = self.obtain_samples(itr=itr, reset_args=[self.reset_arg],preupdate=preupdate)
                       
                   
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    
                    #new_param_values = self.policy.get_variable_values(self.policy.all_params)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]

                    logger.save_itr_params(itr, params, file_name=str(self.reset_arg)+'.pkl')
                    logger.log("Saved")
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)
                    
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    
                    logger.dump_tabular(with_prefix=False)

        if self.log_dir is not None:
            logger.remove_tabular_output(self.log_dir+'/progress.csv')
        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths, comet_logger=self.comet_logger)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

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

    def clip_goal_from_obs(self, paths):
        env = self.env
        while 'clip_goal_from_obs' not in dir(env):
            env = env.wrapped_env
        return env.clip_goal_from_obs(paths)