import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
# from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
import tensorflow as tf

class GaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
            extra_input_dim=0,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape=((2 *(env_spec.observation_space.flat_dim+extra_input_dim) + 3) * num_seq_inputs,),
            output_dim=1,
            name="vf",
            **regressor_args
        )

    @overrides
    def fit(self, paths, log=True):
        # observations = np.concatenate([p["observations"] for p in paths])
        # returns = np.concatenate([p["returns"] for p in paths])
        # self._regressor.fit(observations, returns.reshape((-1, 1)), log=log)

        obs = np.concatenate([np.clip(p["observations"],-10,10) for p in paths])
        obs2 = np.square(obs)
        al = np.concatenate([np.arange(len(p["rewards"])).reshape(-1, 1)/100.0 for p in paths])
        al2 =al**2
        al3 = al**3
        returns = np.concatenate([p["returns"] for p in paths])
        enh_obs = np.concatenate([obs,obs2,al,al2,al3],axis=1)
        # self._regressor._optimizer.reset_optimizer()
        self._regressor.fit(enh_obs, returns.reshape(-1,1), log=log)


        # print("debug11", np.shape(obs))

    @overrides
    def predict(self, path):
        obs = np.clip(path["observations"],-10,10)
        obs2 = np.square(obs)
        al = np.arange(len(path["rewards"])).reshape(-1, 1)/100.0
        al2 =al**2
        al3 = al**3
        enh_obs = np.concatenate([obs,obs2,al,al2,al3],axis=1)
        return self._regressor.predict(enh_obs).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
