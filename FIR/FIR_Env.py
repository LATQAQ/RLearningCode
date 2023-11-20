# 五子棋 强化学习环境

import numpy as np
import pygame as pg
import gymnasium as gym
import threading

Human = 1
Computer = -1
MaxStep = 81
Color = {
    1: (0, 0, 0),
    -1: (255, 255, 255)
}
# state[0]为human, state[1]为computer, state[2]为最后一颗棋子位置

class FIR():
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=np.int8)
        self.winner = 0
        self.done = False
        self.player = 1
        # state 为 己方棋盘、对方棋盘、己方最后一颗棋子位置，对方最后一颗棋子位置
        self.state = np.zeros((4, 9, 9), dtype=np.int8)
        self.reward = 0
        self.action_space = gym.spaces.Discrete(81)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4, 9, 9), dtype=np.int8)

    def reset(self, seed = None):
        info = {}
        self.board = np.zeros((9, 9), dtype=np.int8)
        self.winner = 0
        self.done = False
        self.player = 1
        self.reward = 0
        self.state = np.zeros((4, 9, 9), dtype=np.int8)
        return self.state

    def close(self):
        return

    def seed(self, seed=None):
        return

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def state2str(self, state):
        state_str = ""
        for i in range(9):
            for j in range(9):
                state_str += str(state[0, i, j])
        for i in range(9):
            for j in range(9):
                state_str += str(state[1, i, j])
        for i in range(9):
            for j in range(9):
                state_str += str(state[2, i, j])
        for i in range(9):
            for j in range(9):
                state_str += str(state[3, i, j])
        return state_str

    def getNextState(self, state, action, player):
        next_state = np.copy(state)
        next_state[2, :, :] = 0
        next_state[3, :, :] = 0
        if player == Human:
            next_state[0, action // 9, action % 9] = 1
            next_state[2, action // 9, action % 9] = 1
        else:
            next_state[1, action // 9, action % 9] = 1
            next_state[3, action // 9, action % 9] = 1
        return next_state, -player

    def getGameEnded(self, state, player):
        for i in range(9):
            for j in range(9):
                if state[0, i, j] == 1 and self.check(state, i, j, Human):
                    return Human
                if state[1, i, j] == 1 and self.check(state, i, j, Computer):
                    return Computer
        valids = self.get_valid_actions(self.state2str(state))
        if np.sum(valids) == 0:
            return -1
        return 0

    def check(self, state, x, y, player):
        if player == Human:
            player = 0
        else:
            player = 1
        cnt = 1
        # 五子棋判断胜利
        # 横向
        for i in range(1, 5):
            if y - i >= 0 and state[player, x, y - i] == 1:
                cnt += 1
            else:
                break
        for i in range(1, 5):
            if y + i < 9 and state[player, x, y + i] == 1:
                cnt += 1
            else:
                break
        if cnt >= 5:
            return True
        cnt = 1
        # 纵向
        for i in range(1, 5):
            if x - i >= 0 and state[player, x - i, y] == 1:
                cnt += 1
            else:
                break
        for i in range(1, 5):
            if x + i < 9 and state[player, x + i, y] == 1:
                cnt += 1
            else:
                break
        if cnt >= 5:
            return True
        cnt = 1
        # 斜向
        for i in range(1, 5):
            if x - i >= 0 and y - i >= 0 and state[player, x - i, y - i] == 1:
                cnt += 1
            else:
                break
        for i in range(1, 5):
            if x + i < 9 and y + i < 9 and state[player, x + i, y + i] == 1:
                cnt += 1
            else:
                break
        if cnt >= 5:
            return True
        cnt = 1
        # 反斜向
        for i in range(1, 5):
            if x - i >= 0 and y + i < 9 and state[player, x - i, y + i] == 1:
                cnt += 1
            else:
                break
        for i in range(1, 5):
            if x + i < 9 and y - i >= 0 and state[player, x + i, y - i] == 1:
                cnt += 1
            else:
                break
        return cnt >= 5



    def render(self,state, mode='human'):
        pg.init()
        screen = pg.display.set_mode((900,900))
        pg.display.set_caption("FIR")
        # 填充灰色
        screen.fill((233, 233, 233))
        # 画棋盘
        for i in range(9):
            pg.draw.line(screen, (0, 0, 0), (40 + i * 100, 40), (40 + i * 100, 840), 2)
            pg.draw.line(screen, (0, 0, 0), (40, 40 + i * 100), (840, 40 + i * 100), 2)
        # 画棋子
        for i in range(9):
            for j in range(9):
                if state[0, i, j] == 1:
                    pg.draw.circle(screen, Color[1], (40 + j * 100, 40 + i * 100), 40, 0)
                if state[1, i, j] == 1:
                    pg.draw.circle(screen, Color[-1], (40 + j * 100, 40 + i * 100), 40, 0)
        pg.display.update()

    def mouse_action(self, state):
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                elif event.type == pg.MOUSEBUTTONDOWN:
                    x, y = pg.mouse.get_pos()
                    x = x // 100
                    y = y // 100
                    if state[0, y, x] == 0 and state[1, y, x] == 0:
                        return y * 9 + x


    def getSymmetries(self, state, probs):
        # mirror, rotational
        # probs.shape = (81,)
        new_state = np.copy(state)
        new_action = np.copy(probs)
        new_action = np.reshape(new_action, (9, 9))

        l = []
        for i in range(4):
            new_state = np.rot90(new_state, 1, (1,2))
            new_action = np.rot90(new_action, 1, (0,1))
            r_action = np.reshape(new_action, (81))
            l.append((new_state, r_action))
            new_state = np.flip(new_state, 2)
            new_action = np.flip(new_action, 1)
            r_action = np.reshape(new_action, (81))
            l.append((new_state, r_action))
            # 再翻回来
            new_state = np.flip(new_state, 2)
            new_action = np.flip(new_action, 1)
        return l





    def str2state(self, state_str):
        state = np.zeros((4, 9, 9), dtype=np.int8)
        for i in range(9):
            for j in range(9):
                state[0, i, j] = int(state_str[i * 9 + j])
        for i in range(9):
            for j in range(9):
                state[1, i, j] = int(state_str[i * 9 + j + 81])
        for i in range(9):
            for j in range(9):
                state[2, i, j] = int(state_str[i * 9 + j + 81 * 2])
        for i in range(9):
            for j in range(9):
                state[3, i, j] = int(state_str[i * 9 + j + 81 * 3])
        return state

    def get_valid_actions(self,s):  # 将可下位置置1，不可下位置置0
        state = self.str2state(s)
        valid_actions = np.zeros(81, dtype=np.int8)
        for i in range(9):
            for j in range(9):
                if state[0, i, j] == 0 and state[1, i, j] == 0:
                    valid_actions[i * 9 + j] = 1
        return valid_actions



if __name__ == '__main__':
    # 测试模型，与真人对战
    env = FIR()
    state = env.reset()
    cur_player = 1
    while True:
        env.render(state)
        action = env.mouse_action(state)
        next_state, cur_player = env.getNextState(state, action, cur_player)
        env.render(next_state)
        reward = env.getGameEnded(next_state, cur_player)
        state = next_state
        if reward != 0:
            pg.time.delay(5000)
            pg.quit()
            print('winner: %d' % reward)
            break





