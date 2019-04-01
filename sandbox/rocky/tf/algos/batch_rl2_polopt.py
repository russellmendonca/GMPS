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
from rllab.sampler.utils import rollout, joblib_dump_safe
from maml_examples.maml_experiment_vars import TESTING_ITRS, PLOT_ITRS, VIDEO_ITRS, BASELINE_TRAINING_ITRS


class BatchRL2Polopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with RL^2.
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
            # Note that the number of trajectories for grad update = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            # If default is 10 traj-s, why batch_size=100?
            batch_size=100,
            max_path_length=500,
            meta_batch_size=100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            beta_steps=1,
            beta_curve=None,
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
            post_std_modifier_train=1.0,
            post_std_modifier_test=1.0,
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
        self.beta_steps = beta_steps
        self.beta_curve = beta_curve if beta_curve is not None else [self.beta_steps]
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
        self.use_maml_il = use_maml_il
        self.test_on_training_goals= test_on_training_goals
        self.testing_itrs = TESTING_ITRS
        if self.metalearn_baseline:
            self.testing_itrs.insert(0,0)
        print("test_on_training_goals", self.test_on_training_goals)
        self.limit_demos_num = limit_demos_num
        self.test_goals_mult = test_goals_mult
        self.pre_std_modifier = pre_std_modifier
        self.post_std_modifier_train = post_std_modifier_train
        self.post_std_modifier_test = post_std_modifier_test
        #   self.action_limiter_multiplier = action_limiter_multiplier
        self.expert_trajs_dir = expert_trajs_dir
        self.expert_trajs_suffix = expert_trajs_suffix
        self.use_pooled_goals = use_pooled_goals
        self.extra_input = extra_input
        self.extra_input_dim = extra_input_dim
        # Next, we will set up the goals and potentially trajectories that we plan to use.
        # If we use trajectorie
        assert goals_to_load is None, "deprecated"
        if self.use_pooled_goals:
            if expert_trajs_dir is not None:
                assert goals_pool_to_load is None, "expert_trajs already comes with its own goals, please disable goals_pool_to_load"
                goals_pool = joblib.load(self.expert_trajs_dir+"goals_pool.pkl")
                self.goals_pool = goals_pool['goals_pool']
                self.goals_idxs_for_itr_dict = goals_pool['idxs_dict']
                if "demos_path" in goals_pool.keys():
                    self.demos_path = goals_pool["demos_path"]
                else:
                    self.demos_path = expert_trajs_dir
                print("successfully extracted goals pool", self.goals_idxs_for_itr_dict.keys())
            elif goals_pool_to_load is not None:
                logger.log("Loading goals pool from %s ..." % goals_pool_to_load)
                self.goals_pool = joblib.load(goals_pool_to_load)['goals_pool']
                self.goals_idxs_for_itr_dict = joblib.load(goals_pool_to_load)['idxs_dict']
            else:
                # we build our own goals pool and idxs_dict
                if goals_pool_size is None:
                    self.goals_pool_size = (self.n_itr-self.start_itr)*self.meta_batch_size
                else:
                    self.goals_pool_size = goals_pool_size

                logger.log("Sampling a pool of tasks/goals for this meta-batch...")
                env = self.env
                while 'sample_goals' not in dir(env):
                    env = env.wrapped_env
                self.goals_pool = env.sample_goals(self.goals_pool_size)
                self.goals_idxs_for_itr_dict = {}
                for itr in range(self.start_itr, self.n_itr):
                    self.goals_idxs_for_itr_dict[itr] = rd.sample(range(self.goals_pool_size), self.meta_batch_size)

            # inspecting the goals pool
            env = self.env
            while 'sample_goals' not in dir(env):
                env = env.wrapped_env
            reset_dimensions = env.sample_goals(1).shape[1:]
            dimensions = np.shape(self.goals_pool[self.goals_idxs_for_itr_dict[0][0]])
            assert reset_dimensions == dimensions, "loaded dimensions are %s, do not match with environment's %s" % (
            dimensions, reset_dimensions)
            # inspecting goals_idxs_for_itr_dict
            assert set(range(self.start_itr, self.n_itr)).issubset(set(self.goals_idxs_for_itr_dict.keys())), \
                "Not all meta-iteration numbers have idx_dict in %s" % goals_pool_to_load
            for itr in range(self.start_itr, self.n_itr):
                num_goals = len(self.goals_idxs_for_itr_dict[itr])
                assert num_goals >= self.meta_batch_size, "iteration %s contained %s goals when at least %s are needed" % (itr, num_goals, self.meta_batch_size)
                self.goals_idxs_for_itr_dict[itr] = self.goals_idxs_for_itr_dict[itr][:self.meta_batch_size]

            # we build goals_to_use_dict regardless of how we obtained goals_pool, goals_idx_for_itr_dict
            self.goals_to_use_dict = {}
            for itr in range(self.start_itr, self.n_itr):
                if itr not in self.testing_itrs:
                    self.goals_to_use_dict[itr] = np.array([self.goals_pool[idx] for idx in self.goals_idxs_for_itr_dict[itr]])



        else:  # backwards compatibility code for old-format ETs
            self.goals_to_use_dict = joblib.load(self.expert_trajs_dir+"goals.pkl")

            assert set(range(self.start_itr, self.n_itr)).issubset(set(self.goals_to_use_dict.keys())), "Not all meta-iteration numbers have saved goals in %s" % expert_trajs_dir
        # chopping off unnecessary meta-iterations and goals
            self.goals_to_use_dict = {itr:self.goals_to_use_dict[itr][:self.meta_batch_size]
                                      for itr in range(self.start_itr,self.n_itr)}
        # saving goals pool
        if goals_pickle_to is not None:
            # logger.log("Saving goals to %s..." % goals_pickle_to)
            # joblib_dump_safe(self.goals_to_use_dict, goals_pickle_to)
            logger.log("Saving goals pool to %s..." % goals_pickle_to)
            joblib_dump_safe(dict(goals_pool=self.goals_pool, idxs_dict=self.goals_idxs_for_itr_dict), goals_pickle_to)




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
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix='',testitr=False, preupdate=False):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        paths = self.sampler.obtain_samples(itr=itr, reset_args=reset_args, return_dict=True, log_prefix=log_prefix, extra_input=self.extra_input, extra_input_dim=(self.extra_input_dim if self.extra_input is not None else 0), preupdate=preupdate)
        assert type(paths) == dict
        return paths

    def obtain_agent_info_offpolicy(self, itr, expert_trajs_dir=None, offpol_trajs=None, treat_as_expert_traj=False, log_prefix=''):
        assert expert_trajs_dir is None, "deprecated"
        start = time.time()
        if offpol_trajs is None:
            assert expert_trajs_dir is not None, "neither offpol_trajs nor expert_trajs_dir is provided"
            if self.use_pooled_goals:
                for t, taskidx in enumerate(self.goals_idxs_for_itr_dict[itr]):
                    assert np.array_equal(self.goals_pool[taskidx], self.goals_to_use_dict[itr][t]), "fail"
                offpol_trajs = {t : joblib.load(expert_trajs_dir+str(taskidx)+self.expert_trajs_suffix+".pkl") for t, taskidx in enumerate(self.goals_idxs_for_itr_dict[itr])}
            else:
                offpol_trajs = joblib.load(expert_trajs_dir+str(itr)+self.expert_trajs_suffix+".pkl")

            offpol_trajs = {tasknum:offpol_trajs[tasknum] for tasknum in range(self.meta_batch_size)}

        # some initial rearrangement
        tasknums = offpol_trajs.keys() # tasknums is range(self.meta_batch_size) as can be seen above
        for t in tasknums:
            for path in offpol_trajs[t]:
                if 'expert_actions' not in path.keys() and treat_as_expert_traj:
                   # print("copying expert actions, you should do this only 1x per metaitr")
                    path['expert_actions'] = np.clip(deepcopy(path['actions']), -1.0, 1.0)

                if treat_as_expert_traj:
                    path['agent_infos'] = dict(mean=[[0.0] * len(path['actions'][0])]*len(path['actions']),log_std=[[0.0] * len(path['actions'][0])]*len(path['actions']))
                else:
                    path['agent_infos'] = [None] * len(path['rewards'])

        if not treat_as_expert_traj:
            print("debug12, running offpol on own previous samples")
            running_path_idx = {t: 0 for t in tasknums}
            running_intra_path_idx = {t: 0 for t in tasknums}
            while max([running_path_idx[t] for t in tasknums]) > -0.5: # we cycle until all indices are -1
                observations = [offpol_trajs[t][running_path_idx[t]]['observations'][running_intra_path_idx[t]]
                                for t in tasknums]
                actions, agent_infos = self.policy.get_actions(observations)
                agent_infos = split_tensor_dict_list(agent_infos)
                for t, action, agent_info in zip(itertools.count(), actions, agent_infos):
                    offpol_trajs[t][running_path_idx[t]]['agent_infos'][running_intra_path_idx[t]] = agent_info
                    # INDEX JUGGLING:
                    if -0.5 < running_intra_path_idx[t] < len(offpol_trajs[t][running_path_idx[t]]['rewards'])-1:
                        # if we haven't reached the end:
                        running_intra_path_idx[t] += 1
                    else:

                        if -0.5 < running_path_idx[t] < len(offpol_trajs[t])-1:
                            # we wrap up the agent_infos
                            offpol_trajs[t][running_path_idx[t]]['agent_infos'] = \
                                stack_tensor_dict_list(offpol_trajs[t][running_path_idx[t]]['agent_infos'])
                            # if we haven't reached the last path:
                            running_intra_path_idx[t] = 0
                            running_path_idx[t] += 1
                        elif running_path_idx[t] == len(offpol_trajs[t])-1:
                            offpol_trajs[t][running_path_idx[t]]['agent_infos'] = \
                                stack_tensor_dict_list(offpol_trajs[t][running_path_idx[t]]['agent_infos'])
                            running_intra_path_idx[t] = -1
                            running_path_idx[t] = -1
                        else:
                            # otherwise we set the running index to -1 to signal a stop
                            running_intra_path_idx[t] = -1
                            running_path_idx[t] = -1
        total_time = time.time()-start
       # logger.record_tabular(log_prefix+"TotalExecTime", total_time)
        return offpol_trajs

    def process_samples(self, itr, paths, prefix='', log=True, fast_process=False, testitr=False, metalearn_baseline=False):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log, fast_process=fast_process, testitr=testitr, metalearn_baseline=metalearn_baseline)

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]

        with tf.Session() as sess:
            tf.set_random_seed(1)
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            # sess.run(tf.global_variables_initializer())
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))
            self.start_worker()
            start_time = time.time()
            self.metaitr=0
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                np.random.seed(self.seed+itr)
                tf.set_random_seed(self.seed+itr)
                rd.seed(self.seed+itr)
                with logger.prefix('itr #%d | ' % itr):
                    all_paths_for_plotting = []
                    all_postupdate_paths = []
                    self.beta_steps = min(self.beta_steps, self.beta_curve[min(itr,len(self.beta_curve)-1)])
                    beta_steps_range = range(self.beta_steps) if itr not in self.testing_itrs else range(self.test_goals_mult)
                    beta0_step0_paths = None
                    if self.use_maml_il and itr not in self.testing_itrs:
                        if not self.use_pooled_goals:
                            assert False, "deprecated"
                        else:
                            expert_traj_for_metaitr = {}
                            for t, taskidx in enumerate(self.goals_idxs_for_itr_dict[itr]):
                                demos = joblib.load(self.demos_path+str(taskidx)+self.expert_trajs_suffix+".pkl")
                                # conversion code from Chelsea's format
                                if type(demos) is dict and 'demoU' in demos.keys():
                                    converted_demos = []
                                    for i,demoU in enumerate(demos['demoU']):
                                        converted_demos.append({'observations':demos['demoX'][i],'actions':demoU})
                                    expert_traj_for_metaitr[t] = converted_demos
                                else:
                                    expert_traj_for_metaitr[t] = demos
                            # expert_traj_for_metaitr = {t : joblib.load(self.demos_path+str(taskidx)+self.expert_trajs_suffix+".pkl") for t, taskidx in enumerate(self.goals_idxs_for_itr_dict[itr])}
                        expert_traj_for_metaitr = {t: expert_traj_for_metaitr[t] for t in range(self.meta_batch_size)}

                        # TODO: need to have a middle step for places where demos are saved in demoU and demoX format
                        if self.limit_demos_num is not None:
                            print(self.limit_demos_num)
                            expert_traj_for_metaitr = {t:expert_traj_for_metaitr[t][:self.limit_demos_num] for t in expert_traj_for_metaitr.keys()}
                        for t in expert_traj_for_metaitr.keys():

                            for path in expert_traj_for_metaitr[t]:
                                if 'expert_actions' not in path.keys():
                                    path['expert_actions'] = np.clip(deepcopy(path['actions']), -1.0, 1.0)
                    for beta_step in beta_steps_range:
                        all_samples_data_for_betastep = []
                        self.policy.std_modifier = self.pre_std_modifier
                        self.policy.switch_to_init_dist()  # Switch to pre-update policy
                        if itr in self.testing_itrs:
                            env = self.env
                            while 'sample_goals' not in dir(env):
                                env = env.wrapped_env
                            if self.test_on_training_goals:
                                goals_to_use = self.goals_pool[self.meta_batch_size*beta_step:self.meta_batch_size*(beta_step+1)]
                                print("Debug11", goals_to_use)
                            else:
                                goals_to_use = env.sample_goals(self.meta_batch_size)
                            self.goals_to_use_dict[itr] = goals_to_use if beta_step==0 else np.concatenate((self.goals_to_use_dict[itr],goals_to_use))
                        for step in range(self.num_grad_updates+1): # inner loop
                            logger.log('** Betastep %s ** Step %s **' % (str(beta_step), str(step)))
                            logger.log("Obtaining samples...")

                            if itr in self.testing_itrs:
                                if step < self.num_grad_updates:
                                    print('debug12.0.0, test-time sampling step=', step)
                                    paths = self.obtain_samples(itr=itr, reset_args=goals_to_use,
                                                                    log_prefix=str(beta_step) + "_" + str(step),testitr=True,preupdate=True)
                                    paths = store_agent_infos(paths)  # agent_infos_orig is populated here
                                elif step == self.num_grad_updates:
                                    print('debug12.0.1, test-time sampling step=', step)
                                    paths = self.obtain_samples(itr=itr, reset_args=goals_to_use,
                                                                    log_prefix=str(beta_step) + "_" + str(step),testitr=True,preupdate=False)
                                    all_postupdate_paths.extend(paths.values())
                            elif self.expert_trajs_dir is None or (beta_step == 0 and step < self.num_grad_updates):
                                print("debug12.1, regular sampling")
                                paths = self.obtain_samples(itr=itr, reset_args=self.goals_to_use_dict[itr], log_prefix=str(beta_step)+"_"+str(step),preupdate=True)
                                if beta_step == 0 and step == 0:
                                    paths = store_agent_infos(paths)  # agent_infos_orig is populated here
                                    beta0_step0_paths = deepcopy(paths)
                            elif step == self.num_grad_updates:
                                print("debug12.2, expert traj")
                                paths = self.obtain_agent_info_offpolicy(itr=itr,
                                                                         offpol_trajs=expert_traj_for_metaitr,
                                                                         treat_as_expert_traj=True,
                                                                         log_prefix=str(beta_step)+"_"+str(step))
                            else:
                                assert False, "we shouldn't be able to get here"

                            all_paths_for_plotting.append(paths)
                            logger.log("Processing samples...")
                            samples_data = {}
                            for tasknum in paths.keys():  # the keys are the tasks
                                # don't log because this will spam the console with every task.
                                if self.use_maml_il and step == self.num_grad_updates:
                                    fast_process = True
                                else:
                                    fast_process = False
                                if itr in self.testing_itrs:
                                    testitr = True
                                else:
                                    testitr = False
                                samples_data[tasknum] = self.process_samples(itr, paths[tasknum], log=False, fast_process=fast_process, testitr=testitr, metalearn_baseline=self.metalearn_baseline)

                            all_samples_data_for_betastep.append(samples_data)
                            # for logging purposes only
                            self.process_samples(itr, flatten_list(paths.values()), prefix=str(step), log=True, fast_process=True, testitr=testitr, metalearn_baseline=self.metalearn_baseline)
                            if step == self.num_grad_updates:
                                logger.record_tabular("AverageReturnLastTest", self.sampler.memory["AverageReturnLastTest"],front=True)
                                logger.record_tabular("TestItr", ("1" if testitr else "0"),front=True)
                                logger.record_tabular("MetaItr", self.metaitr,front=True)
                            # logger.log("Logging diagnostics...")
                            # self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))

                            if step < self.num_grad_updates:
                                if itr not in self.testing_itrs:
                                    self.policy.std_modifier = self.post_std_modifier_train*self.policy.std_modifier
                                else:
                                    self.policy.std_modifier = self.post_std_modifier_test*self.policy.std_modifier
                                if (itr in self.testing_itrs or not self.use_maml_il or step<self.num_grad_updates-1) and step < self.num_grad_updates:
                                    # do not update on last grad step, and do not update on second to last step when training MAMLIL
                                    logger.log("Computing policy updates...")
                                    self.policy.compute_updated_dists(samples=samples_data)

                        logger.log("Optimizing policy...")
                        # This needs to take all samples_data so that it can construct graph for meta-optimization.
                        start_loss = self.optimize_policy(itr, all_samples_data_for_betastep)
                        if beta_step == 0 and itr not in self.testing_itrs:
                            print("start loss", start_loss)
                            if self.old_il_loss is not None:
                                if self.old_il_loss < start_loss - 1e-6:
                                    print("reducing betasteps from", self.beta_steps, "to",
                                          int(np.ceil(self.beta_steps / 2)))
                                    self.beta_steps = int(np.ceil(self.beta_steps / 2))
                                self.old_il_loss = min(self.old_il_loss, start_loss)
                            else:
                                self.old_il_loss = start_loss



                    if itr in self.testing_itrs:
                        self.process_samples(itr, flatten_list(all_postupdate_paths), prefix="1",log=True,fast_process=True,testitr=True,metalearn_baseline=self.metalearn_baseline)
                    else:
                        self.metaitr += 1
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data_for_betastep[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data_for_betastep[-1]["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    logger.dump_tabular(with_prefix=False)

                    # The rest is some example plotting code.
                    # Plotting code is useful for visualizing trajectories across a few different tasks.
                    if True and itr in PLOT_ITRS and self.env.observation_space.shape[0] == 2: # point-mass
                        logger.log("Saving visualization of paths")
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            plt.plot(self.goals_to_use_dict[itr][ind][0], self.goals_to_use_dict[itr][ind][1], 'k*', markersize=10)
                            plt.hold(True)

                            preupdate_paths = all_paths_for_plotting[0]
                            postupdate_paths = all_paths_for_plotting[-1]

                            pre_points = preupdate_paths[ind][0]['observations']
                            post_points = postupdate_paths[ind][0]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-b', linewidth=1)

                            pre_points = preupdate_paths[ind][1]['observations']
                            post_points = postupdate_paths[ind][1]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '--r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '--b', linewidth=1)

                            pre_points = preupdate_paths[ind][2]['observations']
                            post_points = postupdate_paths[ind][2]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-.r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-.b', linewidth=1)

                            plt.plot(0,0, 'k.', markersize=5)
                            plt.xlim([-0.8, 0.8])
                            plt.ylim([-0.8, 0.8])
                            plt.legend(['goal', 'preupdate path', 'postupdate path'])
                            plt.savefig(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
                            print(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
                    elif True and itr in PLOT_ITRS and self.env.observation_space.shape[0] == 8:  # reacher
                        logger.log("Saving visualization of paths")

                        # def fingertip(env):
                        #     while 'get_body_com' not in dir(env):
                        #         env = env.wrapped_env
                        #     return env.get_body_com('fingertip')

                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            print("debug13,",itr,ind)
                            a = self.goals_to_use_dict[itr][ind]
                            plt.plot(self.goals_to_use_dict[itr][ind][0], self.goals_to_use_dict[itr][ind][1], 'k*', markersize=10)
                            plt.hold(True)

                            preupdate_paths = all_paths_for_plotting[0]
                            postupdate_paths = all_paths_for_plotting[-1]

                            pre_points = np.array([obs[6:8] for obs in preupdate_paths[ind][0]['observations']])
                            post_points = np.array([obs[6:8] for obs in postupdate_paths[ind][0]['observations']])
                            plt.plot(pre_points[:,0], pre_points[:,1], '-r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-b', linewidth=1)

                            pre_points = np.array([obs[6:8] for obs in preupdate_paths[ind][1]['observations']])
                            post_points = np.array([obs[6:8] for obs in postupdate_paths[ind][1]['observations']])
                            plt.plot(pre_points[:,0], pre_points[:,1], '--r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '--b', linewidth=1)

                            pre_points = np.array([obs[6:8] for obs in preupdate_paths[ind][2]['observations']])
                            post_points = np.array([obs[6:8] for obs in postupdate_paths[ind][2]['observations']])
                            plt.plot(pre_points[:,0], pre_points[:,1], '-.r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-.b', linewidth=1)

                            plt.plot(0,0, 'k.', markersize=5)
                            plt.xlim([-0.25, 0.25])
                            plt.ylim([-0.25, 0.25])
                            plt.legend(['goal', 'preupdate path', 'postupdate path'])
                            plt.savefig(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))
                            print(osp.join(logger.get_snapshot_dir(), 'prepost_path' + str(ind) + '_' + str(itr) + '.png'))

                            if self.make_video and itr in VIDEO_ITRS:
                                logger.log("Saving videos...")
                                self.env.reset(reset_args=self.goals_to_use_dict[itr][ind])
                                video_filename = osp.join(logger.get_snapshot_dir(), 'post_path_%s_%s.mp4' % (ind, itr))
                                rollout(env=self.env, agent=self.policy, max_path_length=self.max_path_length,
                                        animated=True, speedup=2, save_video=True, video_filename=video_filename,
                                        reset_arg=self.goals_to_use_dict[itr][ind],
                                        use_maml=True, maml_task_index=ind,
                                        maml_num_tasks=self.meta_batch_size)
                    elif self.make_video and itr in VIDEO_ITRS:
                        for ind in range(min(5, self.meta_batch_size)):
                            logger.log("Saving videos...")
                            self.env.reset(reset_args=self.goals_to_use_dict[itr][ind])
                            video_filename = osp.join(logger.get_snapshot_dir(), 'post_path_%s_%s.mp4' % (ind, itr))
                            rollout(env=self.env, agent=self.policy, max_path_length=self.max_path_length,
                                    animated=True, speedup=2, save_video=True, video_filename=video_filename,
                                    reset_arg=self.goals_to_use_dict[itr][ind],
                                    use_maml=True, maml_task_index=ind,
                                    maml_num_tasks=self.meta_batch_size)
                        self.policy.switch_to_init_dist()
                        for ind in range(min(5, self.meta_batch_size)):
                            logger.log("Saving videos...")
                            self.env.reset(reset_args=self.goals_to_use_dict[itr][ind])
                            video_filename = osp.join(logger.get_snapshot_dir(), 'pre_path_%s_%s.mp4' % (ind, itr))
                            rollout(env=self.env, agent=self.policy, max_path_length=self.max_path_length,
                                    animated=True, speedup=2, save_video=True, video_filename=video_filename,
                                    reset_arg=self.goals_to_use_dict[itr][ind],
                                    use_maml=False,
                                    # maml_task_index=ind,
                                    # maml_num_tasks=self.meta_batch_size
                                    )
                    elif False and itr in PLOT_ITRS:  # swimmer or cheetah
                        logger.log("Saving visualization of paths")
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            goal_vel = self.goals_to_use_dict[itr][ind]
                            plt.title('Swimmer paths, goal vel='+str(goal_vel))
                            plt.hold(True)

                            prepathobs = all_paths_for_plotting[0][ind][0]['observations']
                            postpathobs = all_paths_for_plotting[-1][ind][0]['observations']
                            plt.plot(prepathobs[:,0], prepathobs[:,1], '-r', linewidth=2)
                            plt.plot(postpathobs[:,0], postpathobs[:,1], '--b', linewidth=1)
                            plt.plot(prepathobs[-1,0], prepathobs[-1,1], 'r*', markersize=10)
                            plt.plot(postpathobs[-1,0], postpathobs[-1,1], 'b*', markersize=10)
                            plt.xlim([-1.0, 5.0])
                            plt.ylim([-1.0, 1.0])

                            plt.legend(['preupdate path', 'postupdate path'], loc=2)
                            plt.savefig(osp.join(logger.get_snapshot_dir(), 'swim1d_prepost_itr' + str(itr) + '_id' + str(ind) + '.pdf'))
        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
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


def store_agent_infos(paths):
    tasknums = paths.keys()
    for t in tasknums:
        for path in paths[t]:
            path['agent_infos_orig'] = deepcopy(path['agent_infos'])
    return paths