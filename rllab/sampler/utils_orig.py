import time

import numpy as np
import joblib
from pathlib import Path

from rllab.misc import tensor_utils


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=False,
            video_filename='sim_out.mp4', reset_arg=None, use_maml=False, maml_task_index=None, maml_num_tasks=None,extra_input_dim=0):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    images = []
    o = env.reset(reset_args=reset_arg)
    if extra_input_dim>0 and use_maml:
        o = np.concatenate((o,[0.0] * extra_input_dim),-1)
    agent.reset()
    path_length = 0
    if animated:
        env1=env
        while hasattr(env1, "wrapped_env"):
            env1 = env1.wrapped_env
        if hasattr(env1, "viewer_setup"):
            env1.viewer_setup()
        env.render()
    while path_length < max_path_length:
        if not use_maml:
            a, agent_info = agent.get_action(observation=o)
        else:
            a, agent_info = agent.get_action_single_env(observation=o, idx=maml_task_index, num_tasks=maml_num_tasks)
        next_o, r, d, env_info = env.step(a)
        if extra_input_dim > 0 and use_maml:
            next_o =np.concatenate((next_o,[0.0]*extra_input_dim),-1)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d: # and not animated:  # TODO testing
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
            if save_video:
                from PIL import Image
                image = env.wrapped_env.wrapped_env.get_viewer().get_image()
                pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                images.append(np.flipud(np.array(pil_image)))

    if animated:
        if save_video and len(images) >= max_path_length:
            import moviepy.editor as mpy
            clip = mpy.ImageSequenceClip(images, fps=10*speedup)
            if video_filename[-3:] == 'gif':
                clip.write_gif(video_filename, fps=10*speedup)
            else:
                clip.write_videofile(video_filename, fps=10*speedup)
        #return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

def joblib_dump_safe(val, filepath):
    # dumps an object making sure we do not overwrite
    assert not Path(filepath).exists(), "cannot overwrite"
    Path(filepath).touch()
    joblib.dump(val, filepath, compress=False)
    return

