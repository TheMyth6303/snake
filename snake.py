from collections import namedtuple
import pygame
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

    def __init__(self, w=640, h=400, block_size=20, speed=10, border=20, num_stones=10):
        self.w = w+2*border
        self.h = h+2*border
        self.border = border
        self.block_size = block_size
        self.snake = None
        self.snake_head = None
        self.snake_direction = None
        self.speed = speed
        self.food = None
        self.stones = []
        self.score = 0
        self.steps = 0
        self.num_stones = num_stones
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake_direction = None
        self.snake_head = Point(self.w/2, self.h/2)
        self.snake = [self.snake_head,
                      Point(self.snake_head.x-self.block_size, self.snake_head.y),
                      Point(self.snake_head.x-(2*self.block_size), self.snake_head.y)]
        self.score = 0
        self.steps = 0
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

    def _update_ui(self):
        self.display.fill(COLORS['stone'])
        pygame.draw.rect(self.display, COLORS['background'],
                         pygame.Rect(self.border, self.border, self.w-2*self.border, self.h-2*self.border)
                         )
        for pt in self.snake:
            pygame.draw.rect(self.display, COLORS['snake_border'],
                             pygame.Rect(pt.x, pt.y, self.block_size, self.block_size)
                             )
            pygame.draw.rect(self.display, COLORS['snake_fill'],
                             pygame.Rect(pt.x+3, pt.y+3, self.block_size-6, self.block_size-6)
                             )
        for stone in self.stones:
            pygame.draw.rect(self.display, COLORS['stone'],
                             pygame.Rect(stone.x, stone.y, self.block_size, self.block_size)
                             )
        pygame.draw.rect(self.display, COLORS['food_border'],
                         pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
                         )
        pygame.draw.rect(self.display, COLORS['food_fill'],
                         pygame.Rect(self.food.x+3, self.food.y+3, self.block_size-6, self.block_size-6)
                         )
        text = font.render("Score: " + str(self.score), True, COLORS['text'])
        self.display.blit(text, [self.border+10, self.border+5])
        pygame.display.flip()

    def play_step(self):
        self.steps += 1
        direction = self.snake_direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    direction = 'l'
                elif event.key == pygame.K_RIGHT:
                    direction = 'r'
                elif event.key == pygame.K_UP:
                    direction = 'u'
                elif event.key == pygame.K_DOWN:
                    direction = 'd'

        self._move(direction)

        over = False
        if self._is_collision():
            over = True
            return over, self.score
        elif self.snake_head == self.food:
            self.score += 1
            self.steps = 1
            self._place_food()
        elif self.snake_direction is not None:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(self.speed)
        if self.steps % 100 == 0:
            self.steps = 0
            self._place_food()
        return over, self.score

    def _move(self, direction):
        x = self.snake_head.x
        y = self.snake_head.y
        if direction == 'r':
            x += self.block_size
        elif direction == 'l':
            x -= self.block_size
        elif direction == 'd':
            y += self.block_size
        elif direction == 'u':
            y -= self.block_size
        if direction is not None:
            if Point(x, y) != self.snake[1]:
                self.snake_head = Point(x, y)
                self.snake.insert(0, self.snake_head)
                self.snake_direction = direction
            else:
                self._move(self.snake_direction)

    def _is_collision(self):
        if self.snake_head in self.snake[1:]:
            return True
        if self.snake_head in self.stones:
            return True
        if self.snake_head.x > self.w - self.border - self.block_size or self.snake_head.x < self.border or \
                self.snake_head.y > self.h - self.border - self.block_size or self.snake_head.y < self.border:
            return True
        return False


if __name__ == '__main__':
    game = Game()

    while True:
        game_over, score = game.play_step()

        if game_over:
            break
    print(f'Your score: {score}')
