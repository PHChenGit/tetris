import numpy as np
import socket
import cv2
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from tetris_env import TetrisEnv
import os
import shutil
import glob
import imageio

ONE_K = 1000
TEN_K = 10 * ONE_K
HUNDRED_K = 100 * ONE_K
ONE_M = 10 * HUNDRED_K 
TEN_M = 10 * ONE_M
HUNDER_M = 100 * ONE_M

csv_file_path = './output/tetris_best_score_a2c_2M.csv'
replay_gif_path = 'replay_a2c.gif'
policy = "CnnPolicy"
save_model_path = f'./models/112598072_a2c_{policy}_30env_2M.zip'
tensorboard_log_path = "./logs/sb3_log"
replay_folder = './replay/A2C'

# Create an environment with 30 client threads
vec_env = make_vec_env(TetrisEnv, n_envs=30)
model = A2C(policy, vec_env, verbose=1, tensorboard_log=tensorboard_log_path)
model.learn(total_timesteps=2*ONE_M, log_interval=50)

# Save model
model.save(save_model_path)

obs = vec_env.reset()
test_steps = 2*HUNDRED_K

if os.path.exists(replay_folder):
    shutil.rmtree(replay_folder)

n_env = obs.shape[0] # Number of environments. A2C will play all envs
ep_id = np.zeros(n_env, int)
ep_steps = np.zeros(n_env, int)
cum_reward = np.zeros(n_env)
max_reward = -1e10
max_game_id = 0
max_ep_id = 0
max_rm_lines = 0
max_lifetime = 0

for step in range(test_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    if step % 50 == 0:
        print(f"Step {step}")
        print("Action: ", action)
        print("reward=", reward, " done=", done)

    for eID in range(n_env):
        cum_reward[eID] += reward[eID]
        folder = f'{replay_folder}/{eID}/{ep_id[eID]}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        fname = folder + '/' + '{:06d}'.format(ep_steps[eID]) + '.png'
        cv2.imwrite(fname, obs[eID])
        #cv2.imshow("Image" + str(eID), obs[eID])
        #cv2.waitKey(10)
        ep_steps[eID] += 1

        if done[eID]:
            if cum_reward[eID] > max_reward:
                max_reward = cum_reward[eID]
                max_game_id = eID
                max_ep_id = ep_id[eID]
                max_rm_lines = info[eID]['removed_lines']
                max_lifetime = info[eID]['lifetime']

            ep_id[eID] += 1
            cum_reward[eID] = 0
            ep_steps[eID] = 0

best_replay_path = f'{replay_folder}/{max_game_id}/{max_ep_id}'


print("After playing 30 envs each for ", test_steps, " steps:")
print(" Max reward=", max_reward, " Best video: " + best_replay_path)
print(" Removed lines=", max_rm_lines, " lifetime=", max_lifetime)

# Write a csv file
with open(csv_file_path, 'w') as fs:
    fs.write('id,removed_lines,played_steps\n')
    fs.write(f'0,{max_rm_lines}, {max_lifetime}\n')
    fs.write(f'1,{max_rm_lines}, {max_lifetime}\n')

# Make a gif image
filenames = sorted(glob.glob(best_replay_path + '/*.png'))

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(replay_gif_path, images, loop=0)
