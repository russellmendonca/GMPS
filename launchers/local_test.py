

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.algos.vpg import VPG as vpg_basic
from sandbox.rocky.tf.algos.vpg_biasADA import VPG as vpg_biasADA
from sandbox.rocky.tf.algos.vpg_fullADA import VPG as vpg_fullADA
from sandbox.rocky.tf.algos.vpg_conv import VPG as vpg_conv

#from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaptivestep_biastransform import MAMLGaussianMLPPolicy as fullAda_Bias_policy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_biasonlyadaptivestep_biastransform import MAMLGaussianMLPPolicy as biasAda_Bias_policy

from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import  SawyerPushEnv 
from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place import SawyerPickPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.door.sawyer_door_open import  SawyerDoorOpenEnv
from multiworld.envs.mujoco.sawyer_xyz.multi_domain.push_door import Sawyer_MultiDomainEnv
from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_coffee import SawyerCoffeeEnv

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.finn_maml_env import FinnMamlEnv
from multiworld.core.wrapper_env import NormalizedBoxEnv
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

import pickle
import argparse
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf
import joblib
import doodad as dd
from doodad.exp_utils import setup

OUTPUT_DIR = '/home/russell/gmps/data/local/'
def experiment(variant):


    seed = variant['seed'] ;  log_dir = variant['log_dir']  ; n_parallel = variant['n_parallel']

    setup(seed, n_parallel , log_dir)

    init_file = variant['init_file'] ; taskIndex = variant['taskIndex'] 
    n_itr = variant['n_itr'] ; default_step = variant['default_step']
    policyType = variant['policyType'] ; envType = variant['envType']

    tasksFile = '/home/russell/multiworld/multiworld/envs/goals/' + variant['tasksFile']+'.pkl'
    tasks = pickle.load(open(tasksFile, 'rb'))

    max_path_length = variant['max_path_length']
 
    use_images = 'conv' in policyType


    if 'MultiDomain' in envType:
        baseEnv = Sawyer_MultiDomainEnv(tasks = tasks , image = use_images , mpl = max_path_length)

    elif 'Push' in envType:   
        baseEnv = SawyerPushEnv(tasks = tasks , image = use_images , mpl = max_path_length)
       

    elif 'PickPlace' in envType:
        baseEnv = SawyerPickPlaceEnv( tasks = tasks , image = use_images , mpl = max_path_length)
       
    elif 'Door' in envType:
        baseEnv = SawyerDoorOpenEnv(tasks = tasks , image = use_images , mpl = max_path_length) 

    elif 'Coffee' in envType:
        baseEnv = SawyerCoffeeEnv(mpl = max_path_length)


    else:
        raise AssertionError('Envs must be Push, PickPlace or Door')

    if use_images:
        obs_keys = ['img_observation']
    else:
        obs_keys = ['state_observation']

   
    env = TfEnv(NormalizedBoxEnv( FinnMamlEnv(FlatGoalEnv(baseEnv, obs_keys))))
    baseline = ZeroBaseline(env_spec=env.spec)
    #baseline = LinearFeatureBaseline(env_spec = env.spec)
    batch_size = variant['batch_size']


    if policyType == 'fullAda_Bias':
    
        baseline = LinearFeatureBaseline(env_spec = env.spec)
        algo = vpg_fullADA(
            env=env,
            policy=None,
            load_policy = init_file,
            baseline=baseline,
            batch_size = batch_size,  # 2x
            max_path_length=max_path_length,
            n_itr=n_itr,
            #noise_opt = True,
            default_step = default_step,
            sampler_cls=VectorizedSampler, # added by RK 6/19
            sampler_args = dict(n_envs=1),
               
            #reset_arg=np.asscalar(taskIndex),
            reset_arg = taskIndex,
            log_dir = log_dir
        )

    elif policyType == 'biasAda_Bias':

        algo = vpg_biasADA(
            env=env,
            policy=None,
            load_policy = init_file, 
            baseline=baseline,
            batch_size= batch_size,  # 2x
            max_path_length=max_path_length,
            n_itr=n_itr,
            #noise_opt = True,
            default_step = default_step,
            sampler_cls=VectorizedSampler, # added by RK 6/19
            sampler_args = dict(n_envs=1),
            #reset_arg=np.asscalar(taskIndex),
            reset_arg = taskIndex,
            log_dir = log_dir
        )

    elif policyType == 'basic':

        algo = vpg_basic(
                env=env,
                policy=None,
                load_policy=init_file,
                baseline=baseline,
                batch_size=batch_size,
                max_path_length=max_path_length,
                n_itr=n_itr,
                #step_size=10.0,
                sampler_cls=VectorizedSampler, # added by RK 6/19
                sampler_args = dict(n_envs=1),
               
                reset_arg=taskIndex,
                optimizer=None,
                optimizer_args={'init_learning_rate': default_step, 'tf_optimizer_args': {'learning_rate': 0.5*default_step}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer},
                log_dir = log_dir
                # extra_input="onehot_exploration", # added by RK 6/19
                # extra_input_dim=5, # added by RK 6/19 
            )


    elif 'conv' in policyType:

        algo = vpg_conv(
            env=env,
            policy=None,
            load_policy = init_file, 
            baseline=baseline,
            batch_size=batch_size,  # 2x
            max_path_length=max_path_length,
            n_itr=n_itr,
            sampler_cls=VectorizedSampler, # added by RK 6/19
            sampler_args = dict(n_envs=1),
            #noise_opt = True,
            default_step = default_step,
            #reset_arg=np.asscalar(taskIndex),
            reset_arg = taskIndex,
            log_dir = log_dir

        )
          
    else:
        raise AssertionError('Policy Type must be fullAda_Bias or biasAda_Bias')

    algo.train()

val = False

#envType = 'SawyerMultiDomain'; max_path_length = 100  ;  tasksFile =  'multi_domain/push_door_v1'
envType = 'Coffee' ; max_path_length = 100 ; tasksFile = 'push_v4'
#envType = 'Push' ; max_path_length = 50 ; tasksFile = 'push_v4_val'

#policyType = 'basic' 
#policyType = 'fullAda_Bias'
policyType = 'biasAda_Bias'
#policyType = 'conv_fcBiasAda'
#policyType = 'conv_no_update'
initFlr = 0.0 ; seed = 0 ; metaFileItr = 15

batch_size = 2000



#initFile = '/home/russell/data/s3/SawyerMultiDomain-Push-Door-v1/gmps/net_100-100-100-100-/policyType_biasAda_Bias/ldim_8/adamSteps_500_mbs_6_fbs_50_initFlr_0.5_seed_1/itr_15.pkl'
#initFile = '/home/russell/doodad/examples/tmp_output/SawyerMultiDomain-Push-Door-v1/gmps/net_100-100-/policyType_fullAda_Bias/ldim_2/adamSteps_500_mbs_3_fbs_20_initFlr_0.5_seed_0/itr_12.pkl'
#expPrefix = 'postSub-Testing-Imitation-valSet-Door-v4-mpl'+str(max_path_length)+'/'+str(policyType)+'_ldim_'+str(ldim)+'_fbs_'+str(fbs)+'_initFlr_'+str(initFlr)+'_metaFileItr_'+str(metaFileItr)+'_mbs_'+str(mbs)+'_adam'+str(adamSteps)+'_bs_'+str(batch_size)+'_seed_'+str(seed)
initFile = '/home/russell/doodad/examples/tmp_output/Coffee-pick_place/gmps/net_150-100-150-/policyType_biasAda_Bias/ldim_2/adamSteps_500_mbs_3_fbs_20_initFlr_0.0_seed_0/itr_9.pkl'


expPrefix = 'SawyerCoffee/'

if 'conv' in policyType:
    expPrefix = 'img-'+expPrefix

for index in range(1):
    expPrefix_numItr = expPrefix+'/Task_'+str(index)+'/'

   
    #for n_itr in range(1,6):
    n_itr = 1
    tf.reset_default_graph()
    expName = expPrefix_numItr+ 'Itr_'+str(n_itr)
    variant = {'taskIndex':index, 'init_file': initFile,  'n_parallel' : 1 ,   'log_dir':OUTPUT_DIR+expName+'/', 'seed' : seed  , 'tasksFile' : tasksFile , 'batch_size' : batch_size,
                    'policyType' : policyType ,  'n_itr' : n_itr , 'default_step' : initFlr , 'envType' : envType , 'max_path_length' : max_path_length}

    experiment(variant)


