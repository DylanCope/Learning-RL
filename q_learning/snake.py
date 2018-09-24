from game import *
import numpy as np
import random
from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN
from pygame import quit as quit_pygame

class Snake(CellGame):
    '''
    '''
    
    def __init__(self, seed=0):
        super(Snake, self).__init__()
        
        self.state_vars = ['snake', 
                           'dir',
                           'next_dir',
                           'snake_len',
                           'berry']
        
        start_x = self.screen_width // 2
        start_y = self.screen_height // 2
        self.state['snake'] = np.array([[start_x, start_y]])
        
        self.state['dir'] = (1, 0)
        self.state['next_dir'] = (1, 0)
        
        self.state['snake_len'] = 1
        
        if seed != 0:
            random.seed(seed)
            
        berry_x = random.randint(0, self.screen_width-1)
        berry_y = random.randint(0, self.screen_height-1)
        self.state['berry'] = np.array([berry_x, berry_y])
        
    def get_controls(self):
        return [self.turn_left,
                self.turn_right,
                self.turn_up,
                self.turn_down]
    
    def turn_left(self):
        ''' Turns the snake to move left.
        '''
        if (self.state['dir'] == np.array([1, 0])).all():
            return False
        self.state['next_dir'] = np.array([-1, 0])
        return True 
     
    def turn_right(self):
        ''' Turns the snake to move right.
        '''
        if (self.state['dir'] == np.array([-1, 0])).all():
            return False
        self.state['next_dir'] = np.array([1, 0])
        return True
     
    def turn_up(self):
        ''' Turns the snake to move up.
        '''
        if (self.state['dir'] == np.array([0, 1])).all():
            return False
        self.state['next_dir'] = np.array([0, -1])
        return True
         
    def turn_down(self):
        ''' Turns the snake to move down.
        '''
        if (self.state['dir'] == np.array([0, -1])).all():
            return False
        self.state['next_dir'] = np.array([0, 1])
        return True
    
    def _move_berry(self):
        berry_x = random.randint(0, self.screen_width-1)
        berry_y = random.randint(0, self.screen_height-1)
        self.state['berry'] = np.array([berry_x, berry_y])

    def _move_snake(self):
        next_head = self.state['snake'][0] + self.state['dir']
        bounds = np.array([self.screen_width, self.screen_height])
        next_head = np.mod(next_head, bounds)
        
        if (next_head == self.state['berry']).all():
            self.state['snake_len'] += 1
            self._move_berry()
            next_rest = self.state['snake']
        else:
            next_rest = self.state['snake'][:-1]
            
        self.state['snake'] = np.vstack((next_head, next_rest))
    
    def _is_colliding_with_self(self):
        head = self.state['snake'][0]
        rest = self.state['snake'][1:]
        match_x = head[0] == rest[:, 0]
        match_y = head[1] == rest[:, 1]
        return (match_x & match_y).any()
    
    def update(self):
        if not self.state['game_over']:
            self.state['dir'] = self.state['next_dir']
            self._move_snake()
            if self._is_colliding_with_self():
                self.end_game()
        super(CellGame, self).update()
        
    def render(self):
        snake_size, _ = self.state['snake'].shape
        for i in range(snake_size):
            self.colour_cell(*self.state['snake'][i])
            
        self.colour_cell(*self.state['berry'], colour = RED)
        
    def score(self):
        return 100 * (self.state['snake_len'] - 1)

class KeyboardUser:
  
    def control(self, snake):
        for event in snake.game_display.pressed_keys:
            if 'key' in event.__dict__:
                if event.key == K_LEFT:
                    snake.turn_left()
                if event.key == K_RIGHT:
                    snake.turn_right()
                if event.key == K_UP:
                    snake.turn_up()
                if event.key == K_DOWN:
                    snake.turn_down()

    def close_game(self):
        quit_pygame()