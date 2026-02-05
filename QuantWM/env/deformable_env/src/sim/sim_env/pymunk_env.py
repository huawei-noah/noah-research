import pygame
from pygame.locals import *
from pygame.color import *

import pyglet
from pyglet.gl import *
from pyglet.window import key, mouse

import pymunk
from pymunk import Vec2d
import pymunk.pyglet_util

import numpy as np
import PIL

class BoxSim(pyglet.window.Window):
    def __init__(self, width, height, box_width, box_height, UPDATE_IMAGE=False):
        pyglet.window.Window.__init__(self, vsync=False)
        
        self.UPDATE_IMAGE = UPDATE_IMAGE    

        # Sim winow parameters. Also define the resolution of the image.
        self.width = width
        self.height = height
        self.set_caption('BoxSim')
        
        # Simulation parameters
        self.space = pymunk.Space()
        
        # cracker bos: (50x158x210) mm
        # sugar box: (28x89x175) mm
        self.box_width = box_width #175
        self.box_height = box_height #89
        
        self.pusher_body = None
        self.velocity = np.array([0., 0.])
        
        self.image = None
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.draw_options.flags = self.draw_options.DRAW_SHAPES
        self.graphics_batch = pyglet.graphics.Batch()
        
        self.global_time = 0.0
        
        # User flags.
        # If this flag is enabled, then the rendering will be done at every
        # simulation timestep. This makes the sim much slower, but is better
        # for visualizing the pusher continuously.
        self.RENDER_EVERY_TIMESTEP = False
        
        self.create_world()
    
    """
    1. Generate environment.
    """
        
    def create_world(self):
        self.space.gravity = Vec2d(0,0) # planar setting
        self.space.damping = 0.0001 # quasi-static. low value is higher damping.
        self.space.color = pygame.color.THECOLORS["white"]
        
        # self.add_walls()
        
        self.wait(1.0) # give some time to stabilize.
        self.render()
    
    def add_walls(self):
        # Create walls
        walls = [
            pymunk.Segment(self.space.static_body, (0, 0), (self.width, 0), 1),  # Top
            pymunk.Segment(self.space.static_body, (0, self.height), (self.width, self.height), 1),  # Bottom
            pymunk.Segment(self.space.static_body, (0, 0), (0, self.height), 1),  # Left
            pymunk.Segment(self.space.static_body, (self.width, 0), (self.width, self.height), 1)  # Right
        ]
        for wall in walls:
            wall.friction = 1.0
            wall.collision_type = 1
            self.space.add(wall)
    
    def add_box(self, center_of_mass=(0,0), friction=0.6):
        self.center_of_mass = center_of_mass
        self.friction = friction
        
        # Create a rectangle box in the middle with an offset center of mass
        self.box_body = pymunk.Body(1.0, 1666) # mass, moment of inertia
        self.box_body.position = (self.width / 2, self.height / 2)
        self.box_body.center_of_gravity = self.center_of_mass
        
        
        self.box_shape = pymunk.Poly.create_box(self.box_body, (self.box_width, self.box_height))
        self.box_shape.friction = self.friction
        self.box_shape.collision_type = 2
        self.box_shape.color = (0, 0, 255, 255) # blue
        
        self.space.add(self.box_body, self.box_shape) 
    
    def add_pusher(self, position):
        self.pusher_body = pymunk.Body(1e7, float('inf'))
        
        self.pusher_body.position = position
        
        pusher_radius = 10 # 1 cm
        self.pusher_shape = pymunk.Circle(self.pusher_body, pusher_radius)
        self.pusher_shape.friction = 0.6
        self.pusher_shape.elasticity = 0.1
        self.pusher_shape.color = (255, 0, 0, 255) # red
        
        self.space.add(self.pusher_body, self.pusher_shape)

    def get_obj_state(self):
        # Get the state of the box.
        # Return: (x, y, theta, dx, dy, dtheta)
        x = self.box_body.position[0]
        y = self.box_body.position[1]
        theta = self.box_body.angle
        dx = self.box_body.velocity[0]
        dy = self.box_body.velocity[1]
        return [x, y, theta, dx, dy]
    
    def get_obj_size(self):
        return [self.box_width, self.box_height]
    
    """
    2. Update and Render Sim. 
    """

    def update(self, u):
        """ Given a control action, run the simulation forward."""
        # Parse into integer coordinates.
        uxf , ufy = u
        
        if self.pusher_body is None:
            self.add_pusher((uxf, ufy))
            self.render()
            return None
        
        uxi, uyi = self.pusher_body.position
        
        # transform into angular coordinates.
        theta = np.arctan2(ufy - uyi, uxf - uxi)
        length = np.linalg.norm(np.array([uxf - uxi, ufy - uyi]), ord=2)
        
        n_sim_step = 60
        step_dt = 1. / n_sim_step
        
        self.velocity = np.array([np.cos(theta), np.sin(theta)]) * length 
        
        for _ in range(n_sim_step):
            self.pusher_body.velocity = self.velocity.tolist()
            self.space.step(step_dt)
            self.global_time += step_dt
        
        self.render()
        return None
        
    
    def wait(self, time):
        """
        Wait for some time in the simulation. Gives some time to stabilize bodies in collision.
        """
        t = 0
        step_dt = 1/60.
        while (t < time):
            self.space.step(step_dt)
            t += step_dt
    
    def render(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT)
        # glLoadIdentity()
        self.space.debug_draw(self.draw_options)
        self.dispatch_events() # necessary to refresh somehow....
        self.flip()

        if self.UPDATE_IMAGE:
            self.update_image()
    
    def update_image(self):
        pitch = -(self.width * len('RGB'))
        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data('RGB', pitch)
        pil_im = PIL.Image.frombytes('RGB', (self.width, self.height), img_data)
        cv_image = np.array(pil_im)[:,:,::-1].copy()
        self.image = cv_image

    def get_current_image(self):
        return self.image
    
    def save_image(self, filename):
        return pyglet.image.get_buffer_manager().get_color_buffer().save(filename)
    
    def close(self):
        pyglet.window.Window.close(self)
        