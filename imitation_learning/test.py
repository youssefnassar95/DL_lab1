from __future__ import print_function

import sys

import torch

sys.path.append("../")

from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        state_gray = rgb2gray(state)

        
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        state_gray = torch.Tensor(state_gray).unsqueeze(0)
        action = agent.predict(state_gray)
        one_hot_encode = np.zeros((1,action.shape[1]))
        max_index = torch.argmax(torch.abs(action))
        one_hot_encode = np.insert(one_hot_encode, max_index, 1)
        one_hot_encode = np.delete(one_hot_encode, max_index+1)
        action_id = ""
        if one_hot_encode[0] == 1.0:
            action_id = STRAIGHT
        elif one_hot_encode[1] == 1.0:
            action_id = LEFT
        elif one_hot_encode[2] == 1.0:
            action_id = RIGHT
        elif one_hot_encode[3] == 1.0:
            action_id = ACCELERATE

        a = id_to_action(action_id)
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = BCAgent()
    agent.load("models/agent1.pt")
    x=0

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
