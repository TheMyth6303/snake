from collections import namedtuple
import pygame
import numpy as np
import random

pygame.init()
font = pygame.font.SysFont('arial', 25)
Point = namedtuple('Point', 'x, y')

COLORS = {'stone': (24, 22, 22),
          'background': (37, 35, 35),
          'snake_fill': (112, 169, 161),
          'snake_border': (64, 121, 140),
          'text': (254, 209, 140),
          'food_fill': (255, 58, 32),
          'food_border': (163, 50, 11)
          }


class Game:

    def __init__(self, w=640, h=400, block_size=20, speed=200, border=20, num_stones=0):
        self.w = w+2*border
        self.h = h+2*border
        self.border = border
        self.block_size = block_size
        self.snake = None
        self.snake_head = None
        self.snake_direction = None
        self.speed = speed
        self.food = None
        self.distance_from_food = None
        self.stones = []
        self.score = 0
        self.steps = 0
        self.num_stones = num_stones
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake_direction = 'r'
        self.snake_head = Point(self.w/2, self.h/2)
        self.snake = [self.snake_head,
                      Point(self.snake_head.x-self.block_size, self.snake_head.y),
                      Point(self.snake_head.x-(2*self.block_size), self.snake_head.y)]
        self.score = 0
        self.steps = 0
        self.stones = []
        self.food = None
        self._place_stones(self.num_stones)
        self._place_food()

    def _place_stones(self, num_stones):
        for stone in range(num_stones):
            x = random.randint(0, (self.w-2*self.border-self.block_size)//self.block_size)*self.block_size + self.border
            y = random.randint(0, (self.h-2*self.border-self.block_size)//self.block_size)*self.block_size + self.border
            stone_position = Point(x, y)
            if stone_position in self.snake+self.stones:
                self._place_stones(num_stones-len(self.stones))
            else:
                self.stones.append(stone_position)

    def _place_food(self):
        x = random.randint(0, (self.w-2*self.border-self.block_size)//self.block_size)*self.block_size + self.border
        y = random.randint(0, (self.h-2*self.border-self.block_size)//self.block_size)*self.block_size + self.border
        self.food = Point(x, y)
        if self.food in self.snake+self.stones:
            self._place_food()
        self.distance_from_food = abs(self.snake_head[0] - self.food[0]) + abs(self.snake_head[0] - self.food[0])

    def _update_ui(self):
        self.display.fill(COLORS['stone'])
        pygame.draw.rect(self.display, COLORS['background'],
                         pygame.Rect(self.border, self.border, self.w - 2 * self.border, self.h - 2 * self.border)
                         )
        for pt in self.snake:
            pygame.draw.rect(self.display, COLORS['snake_border'],
                             pygame.Rect(pt.x, pt.y, self.block_size, self.block_size)
                             )
            pygame.draw.rect(self.display, COLORS['snake_fill'],
                             pygame.Rect(pt.x + 3, pt.y + 3, self.block_size - 6, self.block_size - 6)
                             )
        for stone in self.stones:
            pygame.draw.rect(self.display, COLORS['stone'],
                             pygame.Rect(stone.x, stone.y, self.block_size, self.block_size)
                             )
        pygame.draw.rect(self.display, COLORS['food_border'],
                         pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
                         )
        pygame.draw.rect(self.display, COLORS['food_fill'],
                         pygame.Rect(self.food.x + 3, self.food.y + 3, self.block_size - 6, self.block_size - 6)
                         )
        text = font.render("Score: " + str(self.score), True, COLORS['text'])
        self.display.blit(text, [self.border + 10, self.border + 5])
        pygame.display.flip()

    def play_step(self, action):
        self.steps += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        initial_distance = self.distance_from_food
        self._move(action)
        self.snake.insert(0, self.snake_head)

        over = False
        if self.is_collision():
            over = True
            reward = -20
            return reward, over, self.score
        if self.snake_head == self.food:
            self.score += 1
            self.steps = 0
            reward = 10
            self._place_food()
            return reward, over, self.score
        else:
            if self.distance_from_food > initial_distance:
                reward = -1
            else:
                reward = 1
            self.snake.pop()
        self._update_ui()
        self.clock.tick(self.speed)
        if self.steps % 100 == 0:
            self.steps = 0
            self._place_food()
        return reward, over, self.score

    def _move(self, action):
        if np.array_equal(action, [1, 0, 0]):
            pass
        elif np.array_equal(action, [0, 1, 0]):
            if self.snake_direction == 'r':
                self.snake_direction = 'u'
            elif self.snake_direction == 'u':
                self.snake_direction = 'l'
            elif self.snake_direction == 'l':
                self.snake_direction = 'd'
            else:
                self.snake_direction = 'r'
        else:
            if self.snake_direction == 'r':
                self.snake_direction = 'd'
            elif self.snake_direction == 'd':
                self.snake_direction = 'l'
            elif self.snake_direction == 'l':
                self.snake_direction = 'u'
            else:
                self.snake_direction = 'r'
        x = self.snake_head.x
        y = self.snake_head.y
        if self.snake_direction == 'r':
            x += self.block_size
        elif self.snake_direction == 'l':
            x -= self.block_size
        elif self.snake_direction == 'd':
            y += self.block_size
        elif self.snake_direction == 'u':
            y -= self.block_size
        self.snake_head = Point(x, y)
        self.distance_from_food = abs(self.snake_head[0]-self.food[0]) + abs(self.snake_head[0]-self.food[0])

    def is_collision(self, pt=None):
        if pt:
            point = pt
        else:
            point = self.snake_head
        if point in self.snake[1:]:
            return True
        if point in self.stones:
            return True
        if point.x > self.w - self.border - self.block_size or point.x < self.border or \
                point.y > self.h - self.border - self.block_size or point.y < self.border:
            return True
        return False


if __name__ == '__main__':
    game = Game()
    while True:
        done = game.play_step((1, 0, 0))
        if done[1]:
            break
        done = game.play_step((0, 1, 0))
        if done[1]:
            break
    print('game over')
