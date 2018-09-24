import numpy as np
import pygame
from pygame import *
import sys

BLACK    = (  0,   0,   0)
WHITE    = (255, 255, 255)
DARKGREY = ( 80,  80,  80)
GREY     = (210, 210, 210)
BLUE     = (  0,   0, 255)
GREEN    = (  0, 255,   0)
RED      = (255,   0,   0)
CYAN     = (0  , 255, 255)
ORANGE   = (255, 141,   0)

class Display:

    def __init__(self, dimensions = (100, 100)):
        pygame.init()
        self.dimensions = dimensions
        self.screen = pygame.display.set_mode(dimensions)
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.events = pygame.event.get()
        self.elapsedTime = 0
        self.pressed_keys = []

    def refresh(self):
        self.screen.fill(BLACK)
        self.clock.tick(self.fps)

    def set_dimensions(self, width, height):
        self.dimensions = (width, height)
        self.screen = pygame.display.set_mode(self.dimensions)

    def show(self):
        pygame.display.flip()
        self.elapsedTime += self.get_delta()

    def catch_events(self):
        self.pressed_keys = []
        self.events = pygame.event.get()
        for event in self.events:
            if event.type == KEYDOWN or event.type == QUIT:
                if event.type == QUIT or event.type == K_ESCAPE:
                    pygame.quit()
                self.pressed_keys += [event]

    def get_delta(self):
        return self.clock.get_time() / 1000.0

    def fill(self, colour):
        self.screen.fill(colour)

    def draw_rect(self, x, y, width, height, colour, fill = 0):
        pygame.draw.rect(self.screen, colour, [x, y, width, height], fill)

class Game:

    def __init__(self, dim=(100, 100)):
        self.show = True
        self.game_display = Display(dimensions = dim)
        self.state = { 'game_over' : False }
    
    def start_game(self):
        pass
    
    def update(self):
        if self.show:
            self.game_display.refresh()
            self.render()
            self.game_display.show()
            self.game_display.catch_events()
       
    def render(self):
        pass
    
    def end_game(self):
        self.state['game_over'] = True
    
    def is_game_over(self):
        return self.state['game_over']
    
    def get_controls(self):
        ''' Returns the list of 'buttons' that can be pushed by an agent to change the game state.
            These buttons are Python callables that perform the action.
        '''
        return []
    
class CellGame(Game):
    
    def __init__(self, screen_width=30, screen_height=30,
                 cell_width=10, cell_height=10):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_width = cell_width
        self.cell_height = cell_height
        dim = screen_width*cell_width, screen_height*cell_height
        super(CellGame, self).__init__(dim = dim)
        
    def colour_cell(self, x, y, colour=WHITE):
        self.game_display.draw_rect(x*self.cell_width, 
                                    y*self.cell_height, 
                                    self.cell_width, 
                                    self.cell_height,
                                    colour)
        
