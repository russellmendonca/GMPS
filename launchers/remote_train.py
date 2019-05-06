
from sandbox.rocky.tf.algos.maml_il import MAMLIL

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.maml_gaussian_mlp_baseline import MAMLGaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

#####TODO: combine all these into one policy with different options. Maybe also include MAESN here.
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy as basic_policy
#from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaptivestep import MAMLGaussianMLPPolicy as fullAda_basic_policy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaptivestep_biastransform import MAMLGaussianMLPPolicy as fullAda_Bias_policy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_biasonlyadaptivestep_biastransform import MAMLGaussianMLPPolicy as biasAda_Bias_policy
from sandbox.rocky.tf.policies.maml_minimal_conv_gauss_mlp_policy import MAMLGaussianMLPPolicy as conv_policy
#from sandbox.rocky.tf.policies.maesn_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy as maesn_policy
#from sandbox.rocky.tf.policies.maml_minimal_conv_gauss_mlp_policy import MAMLGaussianMLPPolicy as basic_conv_policy

from sandbox.rocky.tf.optimizers.quad_dist_expert_optimizer import QuadDistExpertOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

from sandbox.rocky.tf.envs.base import TfEnv
# import lasagne.nonlinearities as NL
import sandbox.rocky.tf.core.layers as L

from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import  SawyerPushEnv 
from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place import SawyerPickPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_coffee import SawyerCoffeeEnv
from multiworld.envs.mujoco.sawyer_xyz.door.sawyer_door_open import  SawyerDoorOpenEnv
from multiworld.envs.mujoco.sawyer_xyz.multi_domain.push_door import Sawyer_MultiDomainEnv

from rllab.envs.mujoco.ant_env_rand_goal_ring import AntEnvRandGoalRing
from transferHMS.envs.dclaw.dclaw_screw_rand_goal import DClawScrewRandGoal

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.finn_maml_env import FinnMamlEnv
from multiworld.core.wrapper_env import NormalizedBoxEnv

import tensorflow as tf
import time
from rllab.envs.gym_env import GymEnv

from maml_examples.maml_experiment_vars import MOD_FUNC
import numpy as np
import random as rd
import pickle
import doodad as dd
from doodad.exp_utils import setup


expl = False 
l2loss_std_mult = 0 ; use_corr_term = False

if expl:
	extra_input = "onehot_exploration" ; extra_input_dim = 5
else:
	extra_input =None ; extra_input_dim = 0

beta_steps = 1 ;
meta_step_size = 0.01 ; num_grad_updates = 1
pre_std_modifier = 1.0 ; post_std_modifier = 0.00001 
#limit_demos_num = 5 #pushing
limit_demos_num = None 

test_on_training_goals = True



def experiment(variant):

    seed = variant['seed'] ; n_parallel = 1; log_dir = variant['log_dir']
    setup(seed, n_parallel , log_dir)

    fast_batch_size = variant['fbs']  ; meta_batch_size = variant['mbs']

    adam_steps = variant['adam_steps'] ; max_path_length = variant['max_path_length']

    dagger = variant['dagger'] ; expert_policy_loc = variant['expert_policy_loc']

    ldim = variant['ldim'] ; init_flr =  variant['init_flr'] ; policyType = variant['policyType'] ; use_maesn = variant['use_maesn']
    EXPERT_TRAJ_LOCATION = variant['expertDataLoc']
    envType = variant['envType']

    tasksFile = '/home/code/multiworld/multiworld/envs/goals/' + variant['tasksFile']+'.pkl'
    #asksFile = '/home/russell/multiworld/multiworld/envs/goals/' + variant['tasksFile']+'.pkl'

    all_tasks = pickle.load(open(tasksFile, 'rb'))
    assert meta_batch_size<=len(all_tasks)
    tasks = all_tasks[:meta_batch_size]

    use_images = 'conv' in policyType



    if 'MultiDomain' in envType:
        baseEnv = Sawyer_MultiDomainEnv(tasks = tasks , image = use_images , mpl = max_path_length)

    elif 'Push' == envType:       
        baseEnv = SawyerPushEnv(tasks = tasks , image = use_images , mpl = max_path_length)

    elif envType == 'sparsePush':
        baseEnv = SawyerPushEnv(tasks = tasks , image = use_images , mpl = max_path_length  , rewMode = 'l2Sparse')


    elif 'PickPlace' in envType:
        baseEnv = SawyerPickPlaceEnv( tasks = tasks , image = use_images , mpl = max_path_length)

    elif 'Door' in envType:
        baseEnv = SawyerDoorOpenEnv(tasks = tasks , image = use_images , mpl = max_path_length) 

    elif 'Coffee' in envType:
        baseEnv = SawyerCoffeeEnv(mpl = max_path_length)
        
    elif 'Ant' in envType:
        env = TfEnv(normalize(AntEnvRandGoalRing()))

    elif 'claw' in envType:
        env = TfEnv(DClawScrewRandGoal())

    else:
        assert True == False

    if envType in ['Push' , 'PickPlace' , 'Door' , 'SawyerMultiDomain' , 'Coffee']:
        if use_images:
            obs_keys = ['img_observation']
        else:
            obs_keys = ['state_observation']
        env = TfEnv(NormalizedBoxEnv( FinnMamlEnv(FlatGoalEnv(baseEnv, obs_keys=obs_keys) , reset_mode = 'task')))    

    algoClass = MAMLIL
    baseline = LinearFeatureBaseline(env_spec = env.spec)

    load_policy = variant['load_policy']

    hidden_sizes = variant['hidden_sizes']


    if load_policy !=None:
        policy = None
        load_policy = variant['load_policy']
        if 'conv' in load_policy:
            baseline = ZeroBaseline(env_spec=env.spec)

    elif 'fullAda_Bias' in policyType:
       
        policy = fullAda_Bias_policy(
                name="policy",
                env_spec=env.spec,
                grad_step_size=init_flr,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=hidden_sizes,
                init_flr_full=init_flr,
                latent_dim=ldim
            )

    elif 'biasAda_Bias' in policyType:

        policy = biasAda_Bias_policy(
                name="policy",
                env_spec=env.spec,
                grad_step_size=init_flr,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=hidden_sizes,
                init_flr_full=init_flr,
                latent_dim=ldim
            )

    elif 'basic' in policyType:
        policy =  basic_policy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=init_flr,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=hidden_sizes,                  
        extra_input_dim=(0 if extra_input is "" else extra_input_dim),
    )
   

    elif 'conv' in policyType:

        baseline = ZeroBaseline(env_spec=env.spec)

        policy = conv_policy(
        name="policy",
        latent_dim = ldim,
        policyType = policyType,
        env_spec=env.spec,
        init_flr=init_flr,

        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=hidden_sizes,                 
        extra_input_dim=(0 if extra_input is "" else extra_input_dim),
        )
       

    
    algo = algoClass(
        env=env,
        policy=policy,
        load_policy = load_policy,
        baseline=baseline,
        batch_size=fast_batch_size,  # number of trajs for alpha grad update
        max_path_length=max_path_length,
        meta_batch_size=meta_batch_size,  # number of tasks sampled for beta grad update
        num_grad_updates=num_grad_updates,  # number of alpha grad updates
        n_itr=50, 
        make_video=False,
        use_maml=True,
        use_pooled_goals=True,
        use_corr_term=use_corr_term,
        test_on_training_goals=test_on_training_goals,
        metalearn_baseline=False,
        # metalearn_baseline=False,
        limit_demos_num=limit_demos_num,
        test_goals_mult=1,
        step_size=meta_step_size,
        plot=False,
        beta_steps=beta_steps,
        adam_curve=None,
        adam_steps=adam_steps,
        pre_std_modifier=pre_std_modifier,
        l2loss_std_mult=l2loss_std_mult,
        importance_sampling_modifier=MOD_FUNC[''],
        post_std_modifier = post_std_modifier,
        expert_trajs_dir= EXPERT_TRAJ_LOCATION, 
        expert_trajs_suffix='',
        seed=seed,
        extra_input=extra_input,
        extra_input_dim=(0 if extra_input is "" else extra_input_dim),
        plotDirPrefix = None,
        latent_dim = ldim,
        dagger = dagger , 
        expert_policy_loc = expert_policy_loc
    )
    
    algo.train()


args = dd.get_args()
experiment(args['variant'])
