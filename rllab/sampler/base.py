

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
        self.memory = {}
        self.memory["AverageReturnLastTest"] = 0.0
        self.memory["AverageReturnBestTest"] = 0.0

    def process_samples(self, itr, paths, prefix='', log=True, fast_process=False, testitr=False, metalearn_baseline=False, comet_logger=None):
        baselines = []
        returns = []
        if testitr:
            metalearn_baseline = False
        train_baseline = (itr in BASELINE_TRAINING_ITRS)
        if not fast_process:
            for idx, path in enumerate(paths):
                path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
        if not fast_process and not metalearn_baseline:
            if log:
                logger.log("fitting baseline...")
            if hasattr(self.algo.baseline, 'fit_with_samples'):
                self.algo.baseline.fit_with_samples(paths, samples_data)  # TODO: doesn't seem like this is ever used
            else:
                # print("debug21 baseline before fitting",self.algo.baseline.predict(paths[0])[0:2], "...",self.algo.baseline.predict(paths[0])[-3:-1])
                # print("debug23 predloss before fitting",np.mean([np.mean(np.square(p['returns']-self.algo.baseline.predict(p))) for p in paths]))

                self.algo.baseline.fit(paths, log=log)
                # print("debug25 predloss AFTER  fitting",np.mean([np.mean(np.square(p['returns']-self.algo.baseline.predict(p))) for p in paths]))
                # print("debug22 returns                ",paths[0]['returns'][0:2], "...",paths[0]['returns'][-3:-1])
                # print("debug24 baseline after  fitting",self.algo.baseline.predict(paths[0])[0:2], "...", self.algo.baseline.predict(paths[0])[-3:-1])
            if log:
                logger.log("fitted")

            if 'switch_to_init_dist' in dir(self.algo.baseline):
                self.algo.baseline.switch_to_init_dist()

            if train_baseline:
                self.algo.baseline.fit_train_baseline(paths)

            if hasattr(self.algo.baseline, "predict_n"):
                all_path_baselines = self.algo.baseline.predict_n(paths)
            else:
                all_path_baselines = [self.algo.baseline.predict(path) for path in paths]


        for idx, path in enumerate(paths):
            if not fast_process and not metalearn_baseline:
                # if idx==0:
                    # print("debug22", all_path_baselines[idx])
                    # print("debug23", path['returns'])

                path_baselines = np.append(all_path_baselines[idx], 0)
                deltas = path["rewards"] + \
                         self.algo.discount * path_baselines[1:] - \
                         path_baselines[:-1]
                path["advantages"] = special.discount_cumsum(
                    deltas, self.algo.discount * self.algo.gae_lambda)
                baselines.append(path_baselines[:-1])
            if not fast_process:
                returns.append(path["returns"])
            if "expert_actions" not in path.keys():
                if "expert_actions" in path["env_infos"].keys():
                    path["expert_actions"] = path["env_infos"]["expert_actions"]
                else:
                    # assert False, "you shouldn't need expert_actions"
                    path["expert_actions"] = np.array([[None]*len(path['actions'][0])] * len(path['actions']))


        if not fast_process and not metalearn_baseline: # TODO: we want the ev eventually
            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )
            l2 = np.linalg.norm(np.array(baselines)-np.array(returns))

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])

            if not fast_process:
                rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
                returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])

            
            if "env_infos" in paths[0].keys():
                env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])

            if not fast_process and not metalearn_baseline:
                advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
                # print("debug, advantages are", advantages,)
                # print("debug, shape of advantages is", type(advantages), np.shape(advantages))

            expert_actions = tensor_utils.concat_tensor_list([path["expert_actions"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if not fast_process and not metalearn_baseline:
                if self.algo.center_adv:
                    advantages = util.center_advantages(advantages)
                if self.algo.positive_adv:
                    advantages = util.shift_advantages_to_positive(advantages)
                if "meta_predict" in dir(self.algo.baseline):
                    # print("debug, advantages are", advantages, )
                    advantages = advantages + self.algo.baseline.meta_predict(observations)
                    print("debug, metalearned baseline constant is", self.algo.baseline.meta_predict(observations)[0:2],"...",self.algo.baseline.meta_predict(observations)[-3:-1])
                    # print("debug, metalearned baseline constant shape is", np.shape(self.algo.baseline.meta_predict(observations)))
                # print("debug, advantages are", advantages[0:2],"...", advantages[-3:-1])
                # print("debug, advantages shape is", np.shape(advantages))

            # average_discounted_return = \
            #     np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path.get("rewards",[0])) for path in paths]

            # ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))
            if fast_process:
                samples_data = dict(
                    observations=observations,
                    actions=actions,
                    agent_infos=agent_infos,
                    paths=paths,
                    expert_actions=expert_actions,
                )
            elif metalearn_baseline:
                samples_data = dict(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    returns=returns,
                    agent_infos=agent_infos,
                    paths=paths,
                    expert_actions=expert_actions,
                )
                if 'agent_infos_orig' in paths[0].keys():
                    agent_infos_orig = tensor_utils.concat_tensor_dict_list([path["agent_infos_orig"] for path in paths])
                    samples_data["agent_infos_orig"] = agent_infos_orig
            else:
                samples_data = dict(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    returns=returns,
                    advantages=advantages,
                    env_infos=env_infos,
                    agent_infos=agent_infos,
                    paths=paths,
                    expert_actions=expert_actions,
                )
                if 'agent_infos_orig' in paths[0].keys():
                    agent_infos_orig = tensor_utils.concat_tensor_dict_list([path["agent_infos_orig"] for path in paths])
                    samples_data["agent_infos_orig"] = agent_infos_orig

        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path.get("rewards",[0])) for path in paths]

            # ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        if log and comet_logger:
            comet_logger.log_metric('StdReturn', np.std(undiscounted_returns))
            comet_logger.log_metric('MaxReturn', np.max(undiscounted_returns))
            comet_logger.log_metric('MinReturn', np.min(undiscounted_returns))
            comet_logger.log_metric('AverageReturn', np.mean(undiscounted_returns))
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
            if not fast_process and not metalearn_baseline:
                logger.record_tabular(prefix + 'ExplainedVariance', ev)
                logger.record_tabular(prefix + 'BaselinePredLoss', l2)
                if comet_logger:
                    comet_logger.log_metric('ExplainedVariance', ev)
                    comet_logger.log_metric('BaselinePredLoss', l2)
                # if comet_logger:
                #     comet_logger.log_metric('ExplainedVariance', ev)
                #     comet_logger.log_metric('BaselinePredLoss', l2)

            logger.record_tabular(prefix + 'NumTrajs', len(paths))
            # logger.record_tabular(prefix + 'Entropy', ent)
            # logger.record_tabular(prefix + 'Perplexity', np.exp(ent))
            logger.record_tabular(prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(prefix + 'MinReturn', np.min(undiscounted_returns))
            if "env_infos" in paths[0].keys() and "success_left" in paths[0]["env_infos"].keys():
                logger.record_tabular(prefix + 'success_left', eval_success_left(paths))
                logger.record_tabular(prefix + 'success_right', eval_success_right(paths))
                if comet_logger:
                    comet_logger.log_metric('success_left', eval_success_left(paths))
                    comet_logger.log_metric('success_right', eval_success_right(paths))
            # else:
                # logger.record_tabular(prefix + 'success_left', -1.0)
                # logger.record_tabular(prefix + 'success_right', -1.0)
        # if metalearn_baseline:
        #     if hasattr(self.algo.baseline, "revert"):
        #         self.algo.baseline.revert()

        return samples_data

def eval_success_left(paths):
    if 'env_infos' not in paths[0].keys():
        return float('nan')
    else:
        left_attempts = []
        for path in paths:
            success_left = np.mean(path['env_infos']['success_left'])
            if success_left >= 0:
                left_attempts.append(success_left>0.1)
        return np.mean(left_attempts)


def eval_success_right(paths):
    if 'env_infos' not in paths[0].keys():
        return float('nan')
    else:
        right_attempts = []
        for path in paths:
            success_right = np.mean(path['env_infos']['success_right'])
            if success_right >= 0:
                right_attempts.append(success_right > 0.1)
        return np.mean(right_attempts)


TensorShape=1
OrderedDict=2
Dimension=3

