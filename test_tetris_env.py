from stable_baselines3.common.env_checker import check_env
from tetris_env import TetrisEnv
import matplotlib.pyplot as plt

env = TetrisEnv()
# It will check your custom environment and output additional warnings if needed
# No response may be caused by mismatched action state definition and implementation
print(check_env(env))

obs, info = env.reset()
n_steps = 20
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()  # We render nothing now

    if terminated:
        break

plt.imsave('./test_replay/obs.png', obs)
# plt.imshow(obs)