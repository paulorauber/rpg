import numpy as np
import tensorflow as tf


class Episode:
    def __init__(self, action, observation, reward):
        self.length = 0

        self.actions = []
        self.observations = []
        self.rewards = []

        self.append(action, observation, reward)

    def append(self, action, observation, reward):
        self.length += 1

        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)


class EpisodeVariables:
    def __init__(self, d_observations, d_actions):
        self.d_observations = d_observations
        self.d_actions = d_actions

        self.lengths = tf.placeholder(tf.int32, [None], name='lengths')

        self.actions = tf.placeholder(tf.int32, [None, None], name='actions')
        self.actions_enc = tf.one_hot(self.actions, self.d_actions)

        self.observations = tf.placeholder(tf.float32, [None, None,
                                                        self.d_observations],
                                           name='observations')

        self.rewards = tf.placeholder(tf.float32, [None, None], name='rewards')
        self.rewards_enc = tf.expand_dims(self.rewards, 2)

        self.inputs = [self.actions_enc, self.observations, self.rewards_enc]
        self.inputs = tf.concat(self.inputs, 2)

        self.batch_size = tf.shape(self.inputs)[0]
        self.max_steps = tf.shape(self.inputs)[1]
        self.input_size = tf.shape(self.inputs)[2]


class ValueNetwork:
    def __init__(self, evars, state_size=16):
        self.evars = evars
        self.state_size = state_size

        with tf.variable_scope('value_network'):
            self._model()

        self._loss()

    def _model(self):
        cell = tf.contrib.rnn.LSTMCell(self.state_size)
        self.init_state = cell.zero_state(self.evars.batch_size, tf.float32)

        self.lstm_outputs, \
            self.final_state = tf.nn.dynamic_rnn(cell, self.evars.inputs,
                                                 sequence_length=self.evars.
                                                 lengths,
                                                 initial_state=self.init_state)

        flat = tf.reshape(self.lstm_outputs, [-1, self.state_size])

        self.value = \
            tf.contrib.layers.fully_connected(flat, 1, activation_fn=None)

        self.value = tf.reshape(self.value, [self.evars.batch_size,
                                             self.evars.max_steps])

    def _loss(self):
        rewards_to_go = tf.cumsum(self.evars.rewards, axis=1, exclusive=True,
                                  reverse=True)

        mask = tf.sequence_mask(self.evars.lengths, dtype=tf.float32)

        self.loss = (rewards_to_go - self.value)**2
        self.loss = tf.reduce_sum(self.loss*mask)/tf.reduce_sum(mask)


class PolicyNetwork:
    def __init__(self, evars, state_size=16, value_network=None):
        self.evars = evars
        self.state_size = state_size
        self.value_network = value_network

        with tf.variable_scope('policy_network'):
            self._model()

        self._loss()

    def _model(self):
        cell = tf.contrib.rnn.LSTMCell(self.state_size)
        self.init_state = cell.zero_state(self.evars.batch_size, tf.float32)

        self.lstm_outputs, \
            self.final_state = tf.nn.dynamic_rnn(cell, self.evars.inputs,
                                                 sequence_length=self.evars.
                                                 lengths,
                                                 initial_state=self.init_state)

        flat = tf.reshape(self.lstm_outputs, [-1, self.state_size])

        self.policy = \
            tf.contrib.layers.fully_connected(flat, self.evars.d_actions,
                                              activation_fn=tf.nn.softmax)

        shape = [self.evars.batch_size, self.evars.max_steps,
                 self.evars.d_actions]
        self.policy = tf.reshape(self.policy, shape)

    def _loss(self):
        mask = tf.sequence_mask(self.evars.lengths - 1, dtype=tf.float32)

        actions = self.evars.actions_enc[:, 1:, :]
        policy = self.policy[:, :-1, :]

        log_prob = tf.log(tf.reduce_sum(actions * policy, axis=2))

        advantage = tf.cumsum(self.evars.rewards, axis=1, exclusive=True,
                              reverse=True)[:, :-1]

        if self.value_network is not None:
            ereturn = tf.stop_gradient(self.value_network.value[:, :-1])
            advantage = advantage - ereturn

        self.loss = tf.reduce_sum(log_prob*advantage*mask, axis=1)
        self.loss = -tf.reduce_mean(self.loss)


class Agent:
    def __init__(self, env, state_size=16, batch_size=1, learning_rate=1e-3,
                 use_baseline=True, init_session=True):
        self.env = env
        self.state_size = state_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.d_actions = env.action_space.n
        self.d_observations = env.observation_space.shape[0]

        self.evars = EpisodeVariables(self.d_observations, self.d_actions)

        self.value_net = None
        if use_baseline:
            self.value_net = ValueNetwork(self.evars, state_size)

        self.policy_net = PolicyNetwork(self.evars, state_size, self.value_net)

        if init_session:
            self.init()

    def init(self):
        self.loss = self.policy_net.loss
        if self.value_net is not None:
            self.loss += self.value_net.loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        config = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def interact(self, n_episodes=1, greedy=False, render=False):
        episodes = []

        for i in range(n_episodes):
            self.reset()

            action, observation, reward = -1, self.env.reset(), 0.

            episodes.append(Episode(action, observation, reward))

            if render:
                print('Episode {0}.\n'.format(i + 1))
                self.env.render()

            done = False

            while not done:
                action = self.act(action, observation, reward, greedy)

                observation, reward, done, _ = self.env.step(action)

                episodes[-1].append(action, observation, reward)

                if render:
                    self.env.render()

        return episodes

    def reset(self):
        zeros = np.zeros((1, self.state_size))
        self.policy_state = tf.contrib.rnn.LSTMStateTuple(zeros, zeros)
        self.value_state = tf.contrib.rnn.LSTMStateTuple(zeros, zeros)

    def act(self, action, observation, reward, greedy):
        lengths = np.array([1], dtype=np.int32)
        actions = np.array([[action]], dtype=np.int32)
        observations = np.array([[observation]], dtype=np.float32)
        rewards = np.array([[reward]], dtype=np.float32)

        feed = {self.evars.lengths: lengths, self.evars.actions: actions,
                self.evars.observations: observations,
                self.evars.rewards: rewards,
                self.policy_net.init_state: self.policy_state}

        fetches = [self.policy_net.policy, self.policy_net.final_state]

        if self.value_net is not None:
            feed[self.value_net.init_state] = self.value_state
            fetches.append(self.value_net.final_state)

            policy, self.policy_state, self.value_state = \
                self.session.run(fetches, feed)
        else:
            policy, self.policy_state = self.session.run(fetches, feed)

        if greedy:
            return np.argmax(policy[0][0])

        return np.random.choice(range(self.d_actions), p=policy[0][0])

    def train(self, n_updates, verbose_freq):
        for t in range(1, n_updates + 1):
            episodes = self.interact(self.batch_size, False, render=False)

            lengths = np.array([e.length for e in episodes], dtype=np.int32)
            max_length = max(lengths)

            actions = np.zeros((self.batch_size, max_length), dtype=np.int32)
            observations = np.zeros((self.batch_size, max_length,
                                     self.d_observations), dtype=np.float32)
            rewards = np.zeros((self.batch_size, max_length), dtype=np.float32)

            for i in range(self.batch_size):
                actions[i, :lengths[i]] = episodes[i].actions

                for j in range(lengths[i]):
                    observations[i, j] = episodes[i].observations[j]

                rewards[i, :lengths[i]] = episodes[i].rewards

            feed = {self.evars.lengths: lengths, self.evars.actions: actions,
                    self.evars.observations: observations,
                    self.evars.rewards: rewards}

            loss, _ = self.session.run([self.loss, self.train_step], feed)

            if verbose_freq > 0 and t % verbose_freq == 0:
                aret = rewards.sum(axis=1).mean()
                print('({0}) Mean return: {1}. Loss: {2:f}.'.format(t, aret,
                                                                    loss))
