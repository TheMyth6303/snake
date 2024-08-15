import random
import numpy as np
from snakeRL import Game, Point
from collections import deque
import torch
from helper import plot
from model import QTrainer
from model import QNet

MAX_MEMORY = 100000
BATCHSIZE = 1000


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.memory = deque(maxlen=MAX_MEMORY)  # automatically pops left items after maxlen is reached
        self.model = QNet(20, 32)
        self.trainer = QTrainer(self.model)

    @staticmethod
    def get_state(game):
        # snake direction variables:
        dir_l = int(game.snake_direction == 'l')
        dir_r = int(game.snake_direction == 'r')
        dir_u = int(game.snake_direction == 'u')
        dir_d = int(game.snake_direction == 'd')

        # direction of food drom snake:
        food_l = int(game.food.x < game.snake_head.x)
        food_r = int(game.food.x > game.snake_head.x)
        food_u = int(game.food.y < game.snake_head.y)
        food_d = int(game.food.y > game.snake_head.y)

        match game.snake_direction:
            case 'l':
                straight_pt = Point(game.snake_head.x - game.block_size, game.snake_head.y)
                left_pt = Point(game.snake_head.x, game.snake_head.y + game.block_size)
                right_pt = Point(game.snake_head.x, game.snake_head.y - game.block_size)
            case 'r':
                straight_pt = Point(game.snake_head.x + game.block_size, game.snake_head.y)
                left_pt = Point(game.snake_head.x, game.snake_head.y - game.block_size)
                right_pt = Point(game.snake_head.x, game.snake_head.y + game.block_size)
            case 'u':
                straight_pt = Point(game.snake_head.x, game.snake_head.y - game.block_size)
                left_pt = Point(game.snake_head.x - game.block_size, game.snake_head.y)
                right_pt = Point(game.snake_head.x + game.block_size, game.snake_head.y)
            case _:
                straight_pt = Point(game.snake_head.x, game.snake_head.y + game.block_size)
                left_pt = Point(game.snake_head.x + game.block_size, game.snake_head.y)
                right_pt = Point(game.snake_head.x - game.block_size, game.snake_head.y)

        obs_s = int(game.is_collision(straight_pt))
        obs_l = int(game.is_collision(left_pt))
        obs_r = int(game.is_collision(right_pt))
        dir_tensor = torch.Tensor([dir_l, dir_r, dir_u, dir_d,
                                   food_l, food_r, food_u, food_d,
                                   obs_s, obs_l, obs_r])

        # snake body or stones
        obstacles = torch.zeros((game.h - 2 * game.border) // game.block_size,
                                (game.w - 2 * game.border) // game.block_size)
        for pt in game.snake:
            x = int((pt.x-game.border)//game.block_size)
            y = int((pt.y-game.border)//game.block_size)
            if x == (game.w-2*game.border)//game.block_size:
                obstacles[y][(game.w - 2 * game.border) // game.block_size - 1] = 1.
            elif y == (game.h-2*game.border)//game.block_size:
                obstacles[(game.h - 2 * game.border) // game.block_size - 1][x] = 1.
        x = int((game.snake_head.x-game.border)//game.block_size)
        y = int((game.snake_head.y-game.border)//game.block_size)
        if x == (game.w - 2 * game.border) // game.block_size:
            obstacles[y][(game.w - 2 * game.border) // game.block_size - 1] = 5.
        elif y == (game.h - 2 * game.border) // game.block_size:
            obstacles[(game.h - 2 * game.border) // game.block_size - 1][x] = 5.
        # stones
        for pt in game.stones:
            x = int((pt.x-game.border)//game.block_size)
            y = int((pt.y-game.border)//game.block_size)
            obstacles[y][x] = 1.

        # state: 11 + 20*32 = 651 dimension feature vector
        state = torch.concat([dir_tensor, obstacles.reshape(-1)])
        state = state.view(1, -1)
        return state

    def get_action(self, state):
        self.epsilon = 1-self.n_games/2500
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            predictions = self.model(state)
            move = np.argmax(predictions.detach().numpy())
        final_move[move] = 1
        final_move = torch.Tensor(final_move)
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCHSIZE:
            mini_sample = random.sample(self.memory, BATCHSIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = None, None, None, None, None
        for sample in mini_sample:
            state, action, reward, next_state, done = sample
            if states is None:
                states = state
                actions = action
                rewards = [reward]
                next_states = next_state
                dones = [done]
            else:
                states = torch.concat([states, state])
                actions = torch.concat([actions, action])
                next_states = torch.concat([next_states, next_state])
                rewards.append(reward)
                dones.append(done)
        # states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


def train():
    scores = deque(maxlen=100)
    mean_scores = deque(maxlen=100)
    record = 0
    agent = Agent()
    game = Game()
    while True:
        old_state = agent.get_state(game)
        chosen_move = agent.get_action(old_state)
        reward, game_over, score = game.play_step(chosen_move)
        new_state = agent.get_state(game)
        agent.train_short_memory(old_state, chosen_move, reward, new_state, game_over)
        agent.remember(old_state, chosen_move, reward, new_state, game_over)
        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            scores.append(score)
            total_score = np.array(scores).sum()
            mean_score = total_score / 100
            mean_scores.append(mean_score)
            print('Game:', agent.n_games)
            print('> Current Score:', score)
            print('> Avg(last 100 games):', mean_score)
            print('> Currect Record:', record)
            plot(scores, mean_scores)


if __name__ == '__main__':
    train()
