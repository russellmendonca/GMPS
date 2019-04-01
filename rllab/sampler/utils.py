import numpy as np
from rllab.misc import tensor_utils
import time
import cv2
#from matplotlib import pyplot as plt

def rollout(env, agent, max_path_length=np.inf, speedup=1, save_video=False, video_filename='sim_out.mp4', reset_arg=3 , renderMode = 'human' , return_images = False):
	observations = []
	actions = []
	rewards = []
	agent_infos = []
	env_infos = []
	images = []
	o = env.reset(reset_args=reset_arg)
	
	agent.reset()
	path_length = 0
	
	while path_length < max_path_length:
		a, agent_info = agent.get_action(o)
		next_o, r, d, env_info = env.step(a)
		observations.append(env.observation_space.flatten(o))
		rewards.append(r)
		actions.append(env.action_space.flatten(a))
		agent_infos.append(agent_info)
		env_infos.append(env_info)
		if return_images or save_video:
			images.append(env.render(renderMode))
		path_length += 1
		if d: # and not animated:  # TODO testing
			break
		o = next_o
	
	if save_video:
		import moviepy.editor as mpy
		clip = mpy.ImageSequenceClip(images, fps=20*speedup)
		if video_filename[-3:] == 'gif':
			clip.write_gif(video_filename, fps=20*speedup)
		else:
			clip.write_videofile(video_filename, fps=20*speedup)
	
	
	if return_images:
		return dict(
			observations=tensor_utils.stack_tensor_list(observations),
			actions=tensor_utils.stack_tensor_list(actions),
			rewards=tensor_utils.stack_tensor_list(rewards),
			agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
			env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
			images = np.array(images)
		)
	else:
		return dict(
			observations=tensor_utils.stack_tensor_list(observations),
			actions=tensor_utils.stack_tensor_list(actions),
			rewards=tensor_utils.stack_tensor_list(rewards),
			agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
			env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
		)

