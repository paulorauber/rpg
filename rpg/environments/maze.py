import numpy as np

from gym import Env
from gym import spaces
from gym import Wrapper


class TimedEnvironmentWrapper(Wrapper):
    def __init__(self, env, max_steps):
        Wrapper.__init__(self, env)
        self._max_steps = max_steps

    def reset(self):
        self._n_steps = 0
        return self._reset()

    def step(self, a):
        self._n_steps += 1

        obs, reward, done, info = self._step(a)
        if self._max_steps <= self._n_steps:
            done = True

        return obs, reward, done, info


class Maze(Env):
    def __init__(self, layout, entries, exits, traps=None, can_stay=False,
                 step_reward=-1, stay_reward=-1, exit_reward=10,
                 trap_reward=-100):
        self.metadata = {'render.modes': ['human']}

        self.layout = np.array(layout, dtype=np.int)
        validx, validy = np.nonzero(self.layout)
        self.valid_positions = set(zip(validx, validy))

        self.entries = set(entries)
        self.exits = set(exits)

        self.traps = set()
        if traps is not None:
            self.traps = self.traps.union(traps)

        self.check_consistency()

        self.step_reward = step_reward
        self.stay_reward = stay_reward
        self.exit_reward = exit_reward
        self.trap_reward = trap_reward

        self.action_space = spaces.Discrete(4 + can_stay)
        self.observation_space = None

    def check_consistency(self):
        given_positions = self.entries.union(self.exits, self.traps)

        if not given_positions.issubset(self.valid_positions):
            raise Exception('Invalid entry, exit, or trap.')

        c = len(self.entries) + len(self.exits) + len(self.traps)
        if len(given_positions) < c:
            raise Exception('Two artifacts on same location.')

    def observation(self):
        raise NotImplementedError()

    def _reset(self):
        i = np.random.choice(len(self.entries))
        self.position = sorted(self.entries)[i]

        return self.observation()

    def _step(self, a):
        """a: up, down, left, right, stay"""
        if a >= self.action_space.n:
            raise Exception('Invalid action')

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

        newx = self.position[0] + moves[a][0]
        newy = self.position[1] + moves[a][1]

        if (newx, newy) in self.valid_positions:
            self.position = (newx, newy)

        done = False

        if self.position in self.exits:
            done = True
            reward = self.exit_reward

        if self.position in self.traps:
            done = True
            reward = self.trap_reward

        if not done:
            reward = self.step_reward if a != 4 else self.stay_reward

        return self.observation(), reward, done, {}

    def _render(self, mode='human', close=False):
        if not close:
            print(self.__repr__())

    def __repr__(self):
        s = []

        for i in range(len(self.layout)):
            for j in range(len(self.layout[0])):
                if (i, j) == self.position:
                    s.append('@')
                elif (i, j) in self.exits:
                    s.append('$')
                elif (i, j) in self.traps:
                    s.append('X')
                else:
                    s.append('.' if self.layout[i, j] else '#')
            s.append('\n')

        return ''.join(s)


class MarkovianMaze(Maze):
    def __init__(self, layout, entries, exits, traps=None, can_stay=False,
                 step_reward=-1, stay_reward=-1, exit_reward=10,
                 trap_reward=-100):
        Maze.__init__(self, layout, entries, exits, traps, can_stay,
                      step_reward, stay_reward, exit_reward, trap_reward)

        self.observation_space = spaces.Box(-1, 1, shape=self.layout.size)

    def observation(self):
        obs = np.array(self.layout, dtype=np.float)
        obs[self.position] = -1

        return obs.reshape(-1)


class NonMarkovianMaze(Maze):
    def __init__(self, layout, entries, exits, traps=None, can_stay=False,
                 step_reward=-1, stay_reward=-1, exit_reward=10,
                 trap_reward=-100, color_walls=False):
        Maze.__init__(self, layout, entries, exits, traps, can_stay,
                      step_reward, stay_reward, exit_reward, trap_reward)

        self.observation_space = spaces.Box(0, 1, shape=4)

        if color_walls:
            self.colors = (1 - self.layout)*np.random.random(self.layout.shape)
            self.colors = self.colors*0.5 + self.layout
        else:
            self.colors = np.array(self.layout)

    def observation(self):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        obs = np.zeros(4, dtype=np.float)

        xmax, ymax = self.layout.shape
        for i, move in enumerate(moves):
            newx, newy = self.position[0] + move[0], self.position[1] + move[1]

            if 0 <= newx < xmax and 0 <= newy < ymax:
                obs[i] = self.colors[newx, newy]
            else:
                obs[i] = 0.0

        return obs


def make_random_layout(h, w):
    """Adapted from https://rosettacode.org/wiki/Maze_generation."""
    maze_string = ''

    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["| "] * w + ['|'] for _ in range(h)] + [[]]
    hor = [["+-"] * w + ['+'] for _ in range(h + 1)]

    def walk(x, y):
        vis[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        np.random.shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]:
                continue
            if xx == x:
                hor[max(y, yy)][x] = "+ "
            if yy == y:
                ver[y][max(x, xx)] = "  "
            walk(xx, yy)

    walk(np.random.randint(w), np.random.randint(h))
    for (a, b) in zip(hor, ver):
        maze_string += ''.join(a + ['\n'] + b) + '\n'

    A = [[]]
    for c in maze_string[: -2]:
        if c == '\n':
            A.append([])
        elif c == ' ':
            A[-1].append(1)
        else:
            A[-1].append(0)

    return np.array(A, dtype=np.int)


def make_random_maze(h, w, markovian=True, can_stay=False, step_reward=-1.0,
                     stay_reward=-1.0, exit_reward=None, trap_reward=-10):
    layout = make_random_layout(h, w)
    shape = layout.shape

    MazeClass = MarkovianMaze if markovian else NonMarkovianMaze

    entries = [(1, 1)]
    exits = [(shape[0] - 2, shape[1] - 2)]

    if exit_reward is None:
        exit_reward = 2*(h + w)

    maze = MazeClass(layout, entries, exits, None, can_stay, step_reward,
                     stay_reward, exit_reward, trap_reward)

    return maze


def make_cheese_maze(markovian=True, can_stay=False, step_reward=-1.0,
                     stay_reward=-1.0, exit_reward=5, trap_reward=-10):
    """Adapted from Bakker, Pieter Bram. The state of mind. 2004, pg. 155"""
    layout = np.ones(shape=(3, 5), dtype=np.int)

    layout[1:, 1] = 0
    layout[1:, 3] = 0

    exits = set([(2, 2)])

    entries = set([(2, 0), (2, 4)])

    MazeClass = MarkovianMaze if markovian else NonMarkovianMaze

    maze = MazeClass(layout, entries, exits, None, can_stay, step_reward,
                     stay_reward, exit_reward, trap_reward)

    return maze


def make_wine_maze(markovian=True, can_stay=False, step_reward=-1,
                   stay_reward=-1, exit_reward=8, trap_reward=-10):
    """Adapted from Bakker, Pieter Bram. The state of mind. 2004, pg. 155"""
    layout = np.array([[0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0, 1, 1],
                       [0, 1, 0, 1, 0, 1, 0],
                       [1, 1, 0, 1, 0, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0]], dtype=np.int)

    exits = set([(3, 6)])

    entries = set([(1, 0), (3, 0)])

    MazeClass = MarkovianMaze if markovian else NonMarkovianMaze

    maze = MazeClass(layout, entries, exits, None, can_stay, step_reward,
                     stay_reward, exit_reward, trap_reward)

    return maze


class TMazeWrapper(Wrapper):
    def __init__(self, length, markovian=True, can_stay=False):
        layout = np.zeros(shape=(3, length+1), dtype=np.int)

        layout[:, 0] = 1
        layout[1, :] = 1
        layout[:, -1] = 1

        entries = [(0, 0), (2, 0)]
        exits = [(0, length)]
        traps = [(2, length)]

        MazeClass = MarkovianMaze if markovian else NonMarkovianMaze

        maze = MazeClass(layout, entries, exits, traps, can_stay, -1.0, -1.0,
                         length + 1, -1)

        Wrapper.__init__(self, maze)

    def reset(self):
        self._reset()

        length = self.env.layout.shape[1] - 1

        self.env.exits = set([(0, length)])
        self.env.traps = set([(2, length)])

        if self.env.position == (2, 0):
            self.env.exits, self.env.traps = self.env.traps, self.env.exits

        return self.env.observation()


def make_tmaze(length, markovian=True, can_stay=False):
    return TMazeWrapper(length, markovian, can_stay)


def play(maze, show_observations=True, show_rewards=True):
    udlrx = ['w', 's', 'a', 'd', 'x']

    obs, r, done = maze.reset(), 0., False

    while not done:
        print('State:')
        maze.render()

        if show_observations:
            print('Observation:\n{0}.'.format(obs))
        if show_rewards:
            print('Reward: {0}.'.format(r))

        c = input('Move:')
        if c not in udlrx:
            raise Exception('Invalid action')

        print('')

        obs, r, done, _ = maze.step(udlrx.index(c))

    print('State:')
    maze.render()
    if show_observations:
        print('Observation:\n{0}.'.format(obs))
    if show_rewards:
        print('Reward: {0}.'.format(r))


def main():
    np.random.seed(0)

    maze = make_tmaze(4, markovian=False)
    maze = TimedEnvironmentWrapper(maze, max_steps=10)

    print('Maze:\n')

    play(maze)


if __name__ == "__main__":
    main()
