import gymnasium as gym
from gymnasium import RewardWrapper, wrappers
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from pathlib import Path
import os
import gym_rl_agent as gra


def main():
    env = gym.make("MountainCar-v0")
    sample_states = gra.get_sample_states(env, 1000)
    transformer = gra.Transformer()
    scaler = gra.StandardScaler()
    transformer.add(scaler)
    rbf_samplers = gra.create_rbf_samplers([5.0, 2.0, 1.0, 0.5],[500, 500, 500, 500])
    rbf_union = gra.FeatureUnion([*rbf_samplers])
    transformer.add(rbf_union)
    transformer.fit(sample_states)
    sample_states = transformer.transform(sample_states)

    layers = [
        gra.DenseLayer(3),
    ]

    loss = gra.TDError(gamma=0.9999, lmbda=0.7)

    model = gra.Model(
        layers = layers,
        loss = loss,
        input_data = sample_states,
        learning_rate = 0.1,
        learning_rate_decay = 'inverse',
        decay_rate = 0.01,
        min_learning_rate = 0.01,
    )

    agent = gra.Agent(
        env,
        model,
        transformer,
        epsilon = 0.0,
        min_epsilon = 0.0,
        epsilon_decay=0.005,
        gamma = 0.9999,
        step_reward = None,
        term_reward = None
    )

    for value in agent.train(2000):
            continue


if __name__ == "__main__":
    print('running...')
    main()