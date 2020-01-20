from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
import pickle
import math

def generate_goals(num_goals):
    radius = 2.0 
    angle = np.random.uniform(0, np.pi, size=(num_goals,))
    xpos = radius*np.cos(angle)
    ypos = radius*np.sin(angle)
    return np.concatenate([xpos[:, None], ypos[:, None]], axis=1)

# num_goals = 100
# goals = generate_goals(num_goals)
# import pickle
# pickle.dump(goals, open("goals_ant_val.pkl", "wb"))
# import IPython
# IPython.embed()
class AntEnvRandGoalRing(MujocoEnv, Serializable):

    FILE = 'low_gear_ratio_ant.xml'
    def __init__(self,  num_goals = 40, train = True,  *args, **kwargs):
        
        #self.goals = pickle.load(open('/home/russell/multiworld/multiworld/envs/goals/rad2_quat.pkl' , 'rb')) 
       
        #self.goals = pickle.load(open('/home/code/multiworld/multiworld/envs/goals/rad2_semi.pkl' , 'rb'))

        #thetas = np.linspace( 0, np.pi/2 , 40)[list(range(0,40,4))]
        thetas = np.linspace( 0, np.pi/2 , 40)
        # self.goals = np.array([[2*np.cos(theta) , 2*np.sin(theta)] for theta in thetas])
        self.goals = np.array([[2*np.cos(theta) , 2*np.sin(theta)] for theta in thetas])

        self.goal = None
        self.num_goals = num_goals
        self.sparse = False
        self.info_logKeys = ['goal_dist']
        super(AntEnvRandGoalRing, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

        #self.get_viewer()
        #self.viewer_setup()

    def get_current_obs(self):
       
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            
        ]).reshape(-1)

    def viewer_setup(self):

       self.viewer.cam.trackbodyid = -1
       self.viewer.cam.distance = 6
       self.viewer.cam.azimuth = 90.0
       self.viewer.cam.elevation = -90.0

       self.viewer.cam.lookat[0] = 0
       self.viewer.cam.lookat[1] = 1.2
       self.viewer.cam.lookat[2] = 0

    def sample_goals(self, num_goals):
        return self.goals[np.arange(num_goals)]

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):


        goal_idx = reset_args
        if goal_idx is not None:
            self.goal = self.goals[goal_idx]
        elif self.goal is None:
            self.goal = self.goals[0]

        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    def step(self, action):
        #print(self.sparse)
        #print(self.goals[self._goal_idx])
        self.forward_dynamics(action)
        com = self.get_body_com("torso")
        # ref_x = x + self._init_torso_x

        if self.sparse and np.linalg.norm(com[:2] - self.goal) > 0.8:
            goal_reward = -np.sum(np.abs(self.goal)) + 4.0 
        else:
            goal_reward = -np.sum(np.abs(com[:2] - self.goal)) + 4.0 # make it happy, not suicidal

       
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        infos = {'goal_dist': np.linalg.norm(com[:2] - self.goal) }
        return Step(ob, float(reward), done, **infos)

    @overrides
    def log_diagnostics(self, paths, prefix='', logger = None, comet_logger=None):

        from rllab.misc import logger
        if type(paths[0]) == dict:
            #For SAC
            # for key in self.info_logKeys:
            #     nested_list = [[i[key] for i in paths[j]['env_infos']] for j in range(len(paths))]
            #     logger.record_tabular(prefix + 'last_'+key, np.mean([_list[-1] for _list in nested_list]) )

            #For rllab
            for key in self.info_logKeys:
                # print ("path: ", paths[0])
                logger.record_tabular(prefix + 'last_'+key, np.mean([path['env_infos'][key][-1] for path in paths if 'env_infos' in path]) )

            # if comet_logger:
            #     for key in self.info_logKeys:
            #         # print ("path: ", paths[0])
            #         val = np.mean(
            #             [path['env_infos'][key][-1] for path in paths if 'env_infos' in path])
            #         val = 2.5 if math.isnan(val) else val
            #         comet_logger.log_metric(prefix + 'last_' + key, val)
        else:
            raise NotImplementedError

