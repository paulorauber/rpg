import numpy as np
import tensorflow as tf

from rpg.environments.maze import make_tmaze
from rpg.environments.maze import TimedEnvironmentWrapper

from rpg.core import Agent


def main():
    seed = 0

    corridor_length = 4
    max_steps = 100

    state_size = 16
    batch_size = 1
    learning_rate = 1e-3
    use_baseline = True

    n_train_updates = 10000
    verbose_freq = 100
    n_test_episodes = 5

    np.random.seed(seed)
    tf.set_random_seed(seed)

    maze = make_tmaze(corridor_length, markovian=False)
    maze = TimedEnvironmentWrapper(maze, max_steps=max_steps)

    agent = Agent(maze, state_size, batch_size, learning_rate, use_baseline)

    agent.train(n_train_updates, verbose_freq=verbose_freq)
    agent.interact(n_test_episodes, greedy=True, render=True)
    agent.close()


if __name__ == "__main__":
    main()
