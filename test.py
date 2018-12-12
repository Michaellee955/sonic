import retro
from retro_contest.local import make
import os
import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
import gym_remote.exceptions as gre

from sonic_util import make_env 

# plane游戏
# def main():
#     env = retro.make(game='Airstriker-Genesis', state='Level1')
#     obs = env.reset()
#     while True:
#         obs, rew, done, info = env.step(env.action_space.sample())
#         env.render()
#         if done:
#             obs = env.reset()

# random sonic
# def main():
#     env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
#     obs = env.reset()
#     while True:
#         obs, rew, done, info = env.step(env.action_space.sample())
#         env.render()
#         if done:
#             obs = env.reset()

# right sonic
def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    print('starting episode')
    env.reset()
    while True:
        action = env.action_space.sample()
        action[7] = 1
        ob, reward, done, _ = env.step(action)
        env.render()
        if done:
            print('episode complete')
            env.reset()

if __name__ == '__main__':
    main()