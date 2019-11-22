"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example. The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys

class Ac_Space(object):
    def __init__(self, shape, high, meanings):
        self.shape = [shape]
        self.meanings = meanings
        self.high = high

class Ob_Space(object):
    def __init__(self, shape, data, maze_space):
        self.shape = [shape]
        self.data = data
        self.maze_space = maze_space

class Maze(object):
    def __init__(self, width, height):
        self.action_space = Ac_Space(
            shape=1,
            high=4,
            meanings = {
                0: np.array([-1, 0]),
                1: np.array([1, 0]),
                2: np.array([0, -1]),
                3: np.array([0, 1]),
            }
        )
        self.observation_space = Ob_Space(
            shape=width * height,
            maze_space=[0 for _ in range(width * height)],
            data=[int(p==0) for p in range(width * height)]
        )
        # self.hell = np.array([width - 1, height - 1]) # 终点位置
        self.hell = np.array([0, height - 1]) # 终点位置
        self.oval = np.array([width - 1, 0])  # 陷阱位置
        self.pos = np.array([0, 0])   # 起始位置
        self.x_range = [i for i in range(width)]
        self.y_range = [i for i in range(height)]

    def reset(self):
        self.pos = (0, 0)
        maze_space = self.observation_space.maze_space[::]
        maze_space[0] = 1
        self.observation_space.data = maze_space
        return np.array(maze_space)

    def step(self, action):
        action = int(action[0])
        action = max(action, 0)
        action = min(action, 3)
        delta = self.action_space.meanings[action]
        # 移动
        next_pos = self.pos + delta
        x, y = next_pos
        # 判断是不是碰到边界
        if all((x in self.x_range, y in self.y_range)):
            self.pos = next_pos

        # else:
        #     return self.get_obs(), -1, True, ''
        # 判断状态
        if all(next_pos == self.oval):
            reward = 100
        elif all(next_pos == self.hell):
            reward = -100
        else:
             reward = 0
        return self.get_obs(), reward, reward!=0, ''

    def get_actions(self):
        return self.action_space

    def get_obs(self):
        x,y  = self.pos
        width = len(self.x_range)
        maze_space = self.observation_space.maze_space[::]
        maze_space[x * width + y] = 1
        self.observation_space.data = maze_space
        return np.array(maze_space)

    def render(self):
        width = len(self.x_range)
        data = self.observation_space['data']
        print('^' * 15)
        for idx, d in enumerate(data):
            pos = np.array([idx // width, idx % width])
            if all(pos == self.hell):
                if all(self.pos == self.hell):
                    print('X', end=' ')
                else:
                    print('x', end=' ')
            elif all(pos == self.oval):
                if all(self.pos == self.oval):
                    print('C', end=' ')
                else:
                    print('c', end=' ')
            elif all(pos == self.pos):
                print('·', end=' ')
            else:
                print('o', end=' ')
            if pos[1] == width - 1:
                print('')
        print('-' * 15)


if __name__ == '__main__':
    env = Maze(3, 3)
    env.step(1)
    env.render()
    env.step(1)
    env.render()