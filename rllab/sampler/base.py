

import numpy as np

import rllab.misc.logger as logger
from rllab.algos import util
from rllab.misc import special
from rllab.misc import tensor_utils
from maml_examples.maml_experiment_vars import BASELINE_TRAINING_ITRS


class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError


class BaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.use_prob_latents = self.algo.policy.use_prob_latents
       

    def process_samples(self, itr, paths, prefix='', log=True, task_family_idx = 0 , postUpdate_step=False, testitr=False):
        baselines = [] ; returns = []

        ##################################### Getting Baseline Predictions #######################################
        if not postUpdate_step:
            for idx, path in enumerate(paths):
                path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            if log:
                logger.log("fitting baseline...")

            self.algo.baseline.fit(paths, log=log)
            
            if log:
                logger.log("fitted")

            if hasattr(self.algo.baseline, "predict_n"):
                all_path_baselines = self.algo.baseline.predict_n(paths)
            else:
                all_path_baselines = [self.algo.baseline.predict(path) for path in paths]
        
            for idx, path in enumerate(paths):
               
                path_baselines = np.append(all_path_baselines[idx], 0)
                deltas = path["rewards"] + \
                         self.algo.discount * path_baselines[1:] - \
                         path_baselines[:-1]
                path["advantages"] = special.discount_cumsum(
                    deltas, self.algo.discount * self.algo.gae_lambda)
                baselines.append(path_baselines[:-1])
                returns.append(path["returns"])
        ##############################################################################################################

        ###############################  populating noise for expert traces #################################################
        # if postUpdate_step and not testitr:
        #     noises = 

        ##############################################################################################################

        ##############################   Returning relevant data ####################################################
          
        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
        
        if postUpdate_step and not testitr:
            if self.use_prob_latents:
                #noises = np.zeros((np.shape(observations)[0] , self.latent_dim))
                mpl = int(observations.shape[0]/len(paths))
                all_noises = []
                for path_noise in np.random.normal(0,0.1, size = (len(paths),self.latent_dim)):
                    all_noises.append(np.ones((mpl, self.latent_dim))*path_noise)
            
                noises = np.concatenate(all_noises)
            else:
                noises = np.zeros((np.shape(observations)[0] , self.latent_dim))
        else:
            noises = tensor_utils.concat_tensor_list([path["noises"] for path in paths])
        task_family_idxs = task_family_idx*np.ones((len(noises),), dtype=np.int32)
         
        if postUpdate_step:
            samples_data = dict(
                observations=observations,
                actions=actions,
                agent_infos=agent_infos,
                paths=paths,
                noises = noises,
                task_family_idxs = task_family_idxs
            )
            if not testitr:
                expert_actions = tensor_utils.concat_tensor_list([path["expert_actions"] for path in paths])
                samples_data['expert_actions'] = expert_actions

        elif self.use_prob_latents:

            noises_latent = tensor_utils.concat_tensor_list([path["noises"][0:1] for path in paths])
            task_family_idxs_latent = task_family_idx*np.ones((len(noises_latent),), dtype=np.int32)
            advantages_latent = tensor_utils.concat_tensor_list([path["advantages"][0:1] for path in paths])
            agent_infos_latent = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
            
            if "env_infos" in paths[0].keys():
                env_infos_latent = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])

            if self.algo.center_adv:
                advantages_latent = util.center_advantages(advantages_latent)

            if self.algo.positive_adv:
                advantages_latent = util.shift_advantages_to_positive(advantages_latent)

            samples_data = dict(
                noises=noises_latent,
                task_family_idxs=task_family_idxs_latent,
                advantages=advantages_latent,
                env_infos=env_infos_latent,
                agent_infos=agent_infos_latent,
                paths=paths,
            )

        else:

            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)
            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)
            
           
          
            if "env_infos" in paths[0].keys():
                env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                noises = noises,
                task_family_idxs = task_family_idxs,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )


        if not postUpdate_step and 'dist_infos_orig' in paths[0].keys():
            dist_infos_orig = tensor_utils.concat_tensor_dict_list([path["dist_infos_orig"] for path in paths])
            samples_data["dist_infos_orig"] = dist_infos_orig

        #############################################################################################################################
       

        if log:
            # logger.record_tabular('Iteration', itr)
            # logger.record_tabular('AverageDiscountedReturn',
            #                      average_discounted_return)
            logger.record_tabular(prefix + 'AverageReturn', np.mean(undiscounted_returns))
            if testitr and prefix == "1": # TODO make this functional for more than 1 iteration
                self.memory["AverageReturnLastTest"]=np.mean(undiscounted_returns)
                self.memory["AverageReturnBestTest"]=max(self.memory["AverageReturnLastTest"],self.memory["AverageReturnBestTest"])
                if self.memory["AverageReturnBestTest"] == 0.0:
                    self.memory["AverageReturnBestTest"] = self.memory["AverageReturnLastTest"]
          
            logger.record_tabular(prefix + 'NumTrajs', len(paths))
            logger.record_tabular(prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(prefix + 'MinReturn', np.min(undiscounted_returns))
        
        return samples_data

