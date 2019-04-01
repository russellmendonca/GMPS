import itertools
import pickle

import numpy as np

import rllab.misc.logger as logger
from rllab.misc import tensor_utils
from rllab.sampler.base import BaseSampler
from rllab.sampler.stateful_pool import ProgBarCounter
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.sampler.utils import joblib_dump_safe
from rllab.misc import special


class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None, batch_size=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs
        # if batch_size is not None:
        #     self.batch_size = batch_size
        # else:
        self.batch_size = self.algo.batch_size
    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                #env=pickle.loads(pickle.dumps(self.algo.env)),
                #n = n_envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()


    def obtain_samples(self, itr, reset_args=None, return_dict=False, log_prefix='', extra_input=None, extra_input_dim=None, preupdate=False):
        # reset_args: arguments to pass to the environments to reset
        # return_dict: whether or not to return a dictionary or list form of paths

        logger.log("Obtaining samples for iteration %d..." % itr)

        if extra_input is not None:
            if extra_input == "onehot_exploration":
                if preupdate:
                    print("debug, using extra_input onehot")
                    def expand_obs(obses, path_nums):
                        extra = [special.to_onehot(path_num % extra_input_dim, extra_input_dim) for path_num in path_nums]
                        return np.concatenate((obses, extra), axis=1)
                else:
                    print("debug, using extra_input zeros")
                    def expand_obs(obses, path_nums):
                        extra = [np.zeros(extra_input_dim) for path_num in path_nums]
                        return np.concatenate((obses, extra),axis=1)
            elif extra_input == "onehot_hacked":
                if preupdate:
                    print("debug, using extra_input onehot")
                    def expand_obs(obses, path_nums):
                        extra = [special.to_onehot(3, extra_input_dim) for path_num in path_nums]
                        return np.concatenate((obses, extra), axis=1)
                else:
                    print("debug, using extra_input zeros")
                    def expand_obs(obses, path_nums):
                        extra = [np.zeros(extra_input_dim) for path_num in path_nums]
                        return np.concatenate((obses, extra),axis=1)
            elif extra_input == "gaussian_exploration":
                if preupdate:
                    print("debug, using extra_input gaussian")

                    def expand_obs(obses, path_nums):
                        extra = [np.random.normal(0.,1.,size=(extra_input_dim,)) for path_num in path_nums]
                        return np.concatenate((obses, extra), axis=1)
                else:
                    print("debug, using extra_input zeros")
                    def expand_obs(obses, path_nums):
                        extra = [np.zeros(extra_input_dim) for path_num in path_nums]
                        return np.concatenate((obses, extra), axis=1)


            else:
                def expand_obs(obses, path_nums):
                    return obses
        else:
            def expand_obs(obses, path_nums):
                return obses
        #paths = []
        paths = {}
        for i in range(self.vec_env.num_envs):
            paths[i] = []

        # if the reset args are not list/numpy, we set the same args for each env
        if reset_args is not None and (type(reset_args) != list and type(reset_args) != np.ndarray):
            assert False, "debug, should we be using this?"
            print("WARNING, will vectorize reset_args")
            reset_args = [reset_args]*self.vec_env.num_envs


        n_samples = 0
        path_nums = [0] * self.vec_env.num_envs # keeps track on which rollout we are for each environment instance
        obses = self.vec_env.reset(reset_args)
        obses = expand_obs(obses, path_nums)
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time

        while n_samples < self.batch_size:
            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)
            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, reset_args)   # TODO: instead of receive obs from env, we'll receive it from the policy as a feed_dict
            next_obses = expand_obs(next_obses,path_nums)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths[idx].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])  # TODO: let's also add the incomplete running_paths to paths
                    running_paths[idx] = None
                    path_nums[idx] += 1
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        # adding the incomplete paths
        # for idx in range(self.vec_env.num_envs):
        #     if running_paths[idx] is not None:
        #         paths[idx].append(dict(
        #             observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
        #             actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
        #             rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
        #             env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
        #             agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
        #         ))


        pbar.stop()





      #  logger.record_tabular(log_prefix + "PolicyExecTime", policy_time)
      #  logger.record_tabular(log_prefix + "EnvExecTime", env_time)
       # logger.record_tabular(log_prefix + "ProcessExecTime", process_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())
            #path_keys = flatten_list([[key]*len(paths[key]) for key in paths.keys()])

        return paths
