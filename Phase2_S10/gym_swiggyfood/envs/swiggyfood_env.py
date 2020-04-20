'''
observation Space:
looking ahead of the car 120X40 > Devided into (left | right |front)40x40 > each compressed using maxpool algo to> 20x20
concatinated with car angle between target//goal 

Action Space:
continuous : angles (-5,5)||(-20,20)
'''

import gym
import numpy as np
import skimage.measure
import sys
import copy
import random
import math
from gym import spaces, error
from gym.utils import seeding

import pygame
import pygame.font


from PIL import Image as PILImage
import os
import cv2
import scipy.ndimage


os.environ['SDL_VIDEO_CENTERED'] = "True"


class SwiggyFoodEnv(gym.Env):
    metadata = {'render.modes': ['human', 'console'],
                'video.frames_per_second':100}

    def __init__(self, seed=None):
        # self.seed = seed
        # if seed is None:
        #     self.seed = random.randint(0, sys.maxsize)
        path = './images'
        self.seed()
        
        
        self.bg = pygame.image.load(os.path.join(path, 'citymap.png'))
        self.window_width = 1429
        self.window_height = 660
        # for convolutions//state extraction
        self.sand = np.asarray(PILImage.open(os.path.join(path, "MASK.png")).convert('L'))/255
        self.foodbox = pygame.transform.scale(pygame.image.load(os.path.join(path, 'swiggyfood.png')), (30,30))
        self.delivery = pygame.transform.scale(pygame.image.load(os.path.join(path, 'delivery.png')), (50,50))


        #spawn car
        self.car_img = pygame.transform.scale(pygame.image.load(os.path.join(path, 'car.png')), (20,10))
        self.car = self.car_img.get_rect()        
        self.car.centerx = self.window_width//2
        self.car.centery = self.window_height//2
        self.car.width = 20
        self.car.height = 10
        self.speed = 3  # in pixels

        self.angle = 0  # car's angle from x axis, degree for now      
        
        # self.goal_x = 163 # x-coordinate of the goal
        # self.goal_y = 539# y-coordinate of the goal
        
        self.last_distance = 0
        self.last_reward = 0
        self.swap = 0
               
        # 'inspired from gym env'
        self.max_episode_steps = 2000

        
        self.max_action = 10 #action angle in degree
        self.action_space = spaces.Box(-self.max_action, self.max_action, shape=(1,), dtype=np.int8)

        self.state_size = 60 #cutout for convolutions

        self.observation_space = spaces.Dict({"surround": spaces.Box(low=0, high=1, shape=(self.state_size, self.state_size, 1), dtype=np.float64),
                                              "orientation": spaces.Discrete(3,)})
        self.reset()

    def reset(self):
        """
        This function resets the environment and also returns the init. state.
        """
        #Get Pickup A and  Delivery B
        y, x = np.where(self.sand == 0) # pixels of the road
        
        #spawning the car
        k = self.np_random.randint(len(x))
        self.car.centerx, self.car.centery = int(x[k]), int(y[k])
        # self.car.centerx = self.np_random.randint(12, 1417)
        # self.car.centery = self.np_random.randint(12, 610)
        
        # Randomly create pickup and drop points on the road
        # i = self.np_random.randint(len(x))
        # j = self.np_random.randint(len(x))
        # #Pickup : A foodbox
        self.x1, self.y1 = int(x[i]), int(y[i])
        # self.x1, self.y1 = int(16), int(575)
        # #Delivery : B
        self.x2, self.y2 = int(x[j]), int(y[j])
        # self.x2, self.y2 = int(1420), int(36) 

        # Fist Target A
        self.goal_x, self.goal_y = self.x1, self.y1
        self.swap = 0   # resetting the destinations
        
        self.angle = 0  # car's angle from x axis, degree for now
        self.rotation = 0 #rotation// the action// just the initialization
        self.speed = 3  # in pixels

        # for rendering; most of the stuff is same,     
        self.screen = None
        self.surr = pygame.surfarray.make_surface(np.zeros((60,60)))

        return self._get_state()

    def render(self, mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        TODO : render using default gym functions

        """
        if mode == 'console':
            print(self._get_state)
        elif mode == "human":
            try:
                import pygame
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: pygame install karo mere bhai; use `pip install pygame`".format(e))
            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode(
                        (round(self.window_width), round(self.window_height)))
                    # surr = self.surr
                    
                clock = pygame.time.Clock()
                #trick to keep the screen running:
                pygame.event.pump()
                                
                self.screen.blit(self.bg, (0,0))

                # Draw car running
                # TODO : change the car orientation and better resize perhapes
                car_img_angle = pygame.transform.rotate(self.car_img, -self.angle)
                self.screen.blit(car_img_angle, (self.car))

                # show foodbox and delivery
                self.screen.blit(self.foodbox, (self.x1-30, self.y1-30))
                self.screen.blit(self.delivery, (self.x2-50, self.y2-50))

                self.screen.blit(self.surr, (1140,500))


                pygame.display.update()
                clock.tick(self.metadata["video.frames_per_second"])
                
                #trick to quit the screen
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

                
        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)

    def step(self, action):
        """
        This method steps the game forward one step and moves the car one step = [x,y]+vector(speed, angle)

        Parameters
        ----------
        action : int
            The action is an angle between -20 and 20 degrees,
            decides new angle of the car;
            angle = angle + action

        Returns
        -------
        ob, reward, episode_over/done, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.here is's a dictionary
                "surroud": 40x40 grid,
                "orientation": (1)Direction of the car with respect to the goal (in + and -)
                                (if the car is heading perfectly towards the goal, then orientation = 0)
                                possibly calculated by Vector(*self.car.velocity).angle((xx,yy))/180
                                (2) distance between car and goal
                                [orientation, -orientation, distance]


            reward (float) :(maybe int)
                amount of reward achieved by the previous action.
            episode_over (bool) : done
                whether it's time to reset the environment again.
            info (dict) :
                Kuch ankahi si baatein. :'D
        """
     
        # move the freakin car
        self.angle, self.car.centerx, self.car.centery = self._move_car(action)
        reward, done = self._reward_calc()
        state = self._get_state()
        self.surr = pygame.surfarray.make_surface(state["surround"].reshape(60,60)*255)

        return state, reward, done, {}
    
    def _reward_calc(self, reward = 0, done = False):
        """
        !  MOST IMPORTNANT PART !  
        TODO: too many ifs and buts// Use some Math
        """
        # Calculate the distance after moving
        distance = np.sqrt((self.car.centerx - self.goal_x)**2 + (self.car.centery - self.goal_y)**2) # getting the new distance between the car and the goal right after the car moved
        #let's get to rewards// make them complicated
        if self.sand[int(self.car.centery), int(self.car.centerx)] > 0: # if the car is on the sand
            self.speed = 3  # in pixels
            # reward = -1
            reward = -0.85 if distance < self.last_distance else -1 # and reward = -1
            
        
        else: # being on road
            self.speed = 5 # car runs to a normal speed
            # reward = 0.3 # reward for coming on Road
            reward = self.last_reward + 0.2 if self.last_reward >= 0.3 else 0.3 # if it was on road last time or not
            if self.last_reward >= -0.5: # punishment or reward for moving toward goal
                reward = self.last_reward - 0.5 if distance >= self.last_distance else self.last_reward + 0.5 
        
        reward = 0.9 if reward >= 0.9 else reward # max reward is 0.9
        reward = -1 if reward <= -1 else reward # min reward is -1


        # boundary control because why not!!
        # TODO: modify reward function as such to avoid getting stuck to wall || DONE
        # if self.car.centerx < 10: # if the car is in the left edge of the frame
        #     reward =  -1 # but it gets bad reward -1
        # elif self.car.centerx > self.window_width-10: # if the car is in the right edge of the frame
        #     reward = -1 # but it gets bad reward -1
        # if self.car.centery < 10: # if the car is at top edge of the frame
        #     reward = -1 # but it gets bad reward -1
        # elif self.car.centery > self.window_height-10: # if the car is in the bottom edge of the frame
        #     reward = -1 # but it gets bad reward -1

        # Point A as goal has been given in Reset
        if distance < 30 and not done: #when reaches somewhere
            if self.swap==1:
                # reward = 2
                done = True
            else: # reached A
                self.goal_x, self.goal_y = self.x2, self.y2
                # self.x1, self.y1 = self.car.centerx-10, self.car.centery-10 # foodbox attached to car
                # reward = 2
                self.swap = 1

        #take the foodbox/picked up box with car
        if self.swap == 1:
            self.x1, self.y1 = self.car.centerx-10, self.car.centery-10 # foodbox attached to car
        
        # Updating the last distance from the car to the goal and reward
        self.last_distance = distance
        self.last_reward = reward

        return reward, done

    def _move_car(self, action):
        """
        Moves the car forward.
        action is rotation of the car
        """
        # update the freaking angle first
        # angle, speed = action

        self.angle = (self.angle + action)%360
        # self.angle = (self.angle + action[0])%360
        # print(car.centerx, car.centery)
        #get the new x, y
        self.car.centerx, self.car.centery = self._new_xy(self.car.centerx,
                                                          self.car.centery,
                                                          self.speed,
                                                          # action[1]//2, # hack to clip speed between [-10, 10]
                                                          # action[1],
                                                          self.angle)
        # print(car.centerx,car.centery)

        # Collision control and not letting car move out of boundary
        # collision with left wall
        if self.car.centerx  < 15:
            self.car.centerx = 15
        # collision with right wall
        if self.car.centerx >= self.window_width-15:
            self.car.centerx = self.window_width-15
        # collision with top wall
        if self.car.centery < 15:
            self.car.centery = 15
        # collision with bottom wall
        elif self.car.centery >= self.window_height-15:
            self.car.centery = self.window_height-15
        return self.angle, self.car.centerx, self.car.centery

    def _new_xy(self, x, y, speed, angle):
        """
        new position of car after one timestamp and single velocity value
        """
        new_x = x + int(speed * math.cos(math.radians(angle)))
        new_y = y + int(speed * math.sin(math.radians(angle)))
        # print(new_x, new_y)
        return new_x, new_y

    def _get_state(self):
        """
        This function returns the current game state.
        state["surround"] = 60x60 
        spaces.Box(low=0, high=1, shape=(screen_height, screen_width, 1), dtype=np.uint8)

        state["orientation"] = angle between car position and goal, distance, and some more if needed later
        """
        state = {}
      
        crop_size = 60
        cutout = self._subimage(crop_size=crop_size)
        state['surround'] = skimage.measure.block_reduce(cutout, (1,1,1), np.max) # makes it (crop_size//2)x(crop_size//2)

        # direction of the car with respect to the goal (if the car is heading perfectly towards the goal, then orientation = 0)
        xx = self.goal_x - self.car.centerx # difference of x-coordinates between the goal and the car
        yy = self.goal_y - self.car.centery # difference of y-coordinates between the goal and the car
        orientation = math.degrees(math.atan2(yy, xx))%360 # orientation from goal

        distance = np.sqrt((self.car.centerx - self.goal_x)**2 + (self.car.centery - self.goal_y)**2) # getting the new distance between the car and the goal right after the car moved
        # distance_st = (self.last_distance-distance)/distance

        # state["orientation"] = np.asarray([orientation,-orientation, self.angle, -self.angle, self.speed], dtype = np.float32)
        state["orientation"] = np.asarray([distance, orientation,-orientation], dtype = np.float32)
        return state


    def _subimage(self, crop_size=80): # crop size will later be 2x2 maxpool to make 40x40 or not 
        #120 padding to avoid out of limit cropping at edges
        pad = 120
        crop1 = np.pad(self.sand, pad_width=pad, mode='maximum') # maximum is 1 in the mask1
        centerx = self.car.centerx + pad
        centery = self.car.centery + pad

        #smaller cutout to rotate and save memory 
        startx = int(centerx-(crop_size*2))
        starty = int(centery-(crop_size*2))
        crop1 = crop1[starty:starty+crop_size*4, startx:startx+crop_size*4]

        # if theta ko radians ki aag lagi ho
        #theta = self.angle*180/np.pi
        # print(crop1.shape, self.car.centerx, self.car.centery)

        shape = ( crop1.shape[1], crop1.shape[0] ) # cv2.warpAffine expects shape in (length, height)
        center = (crop1.shape[1]//2, crop1.shape[0]//2)
        matrix = cv2.getRotationMatrix2D( center=center, angle= -self.angle, scale=1 )
        crop1 = cv2.warpAffine( src=crop1, M=matrix, dsize=shape )
        crop1 = crop1.clip(min=0) # remove negative number due to interpolation for rotation
        center = (crop1.shape[1]//2, crop1.shape[0]//2)

        # x = int( center[0] - crop_size//6) # helps us with 1/3 back and 2/3 front
        x = int( center[0] - crop_size//2)
        y = int( center[1] - crop_size//2 ) # center-60 up

        crop1 = crop1[ y:y+crop_size, x:x+crop_size ]
        crop1 = np.rot90(crop1,2) # car dash view toward north
        return crop1.reshape(crop_size, crop_size, 1)
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# Running the whole thing just to diagnose
# if __name__ == '__main__':
#     # state = SwiggyFoodEnv().reset()
# #     print(state)
# #     print(state.shape)
# #     print(state)
#     # print(SwiggyFoodEnv().action_space[0])
# #     # orientation
#     a = SwiggyFoodEnv()
#     a.reset()
#     print(a.x1,a.y1,a.x2,a.y2 )
#     b = SwiggyFoodEnv()
#     b.reset()
#     print(b.x1,b.y1,b.x2,b.y2 )
    
#     # from matplotlib import pyplot as plt
#     # plt.imshow(state.reshape(40,40), interpolation=None)
#     # plt.show()
