import math
import pygame
import time
import random
import numpy as np
import copy
import gym
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pickle as pkl
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Vector2D:
    def __init__(self, *args):
        if isinstance(args[0], tuple):
            self.x, self.y = args[0]
        else:
            self.x, self.y = args

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        if isinstance(other, Vector2D):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Vector2D):
            return self.x != other.x or self.y != other.y
        return NotImplemented

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)

    def __iter__(self):
        return iter((self.x, self.y))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def rotate(self, angle):
        radians = math.radians(angle)
        cos_val = math.cos(radians)
        sin_val = math.sin(radians)
        x = self.x * cos_val - self.y * sin_val
        y = self.x * sin_val + self.y * cos_val
        return Vector2D(x, y)

    def angle_to(self, other):
        # Calculate the angle between self and other
        angle = math.degrees(math.atan2(other.y - self.y, other.x - self.x))

        # Return a rotated vector pointing in the direction of other
        return angle

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def normalize(self):
        length = self.length()
        if length > 0:
            return self / length
        return Vector2D(0, 0)

    @staticmethod
    def remap_scalar(x, old_min, old_max, new_min, new_max):
        return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class Lander:
    def __init__(self, init_position, init_velocity):
        # Constants
        self.GRAVITY = 3.711
        self.LANDER_RADIUS = 50
        self.LANDING_LEG_ANGLE = 25
        # from the center of the body
        self.LANDING_LEG_LENGTH = 100

        self.crashed = False
        self.landed = False
        self.body_collision = False
        self.left_leg_collision = False
        self.right_leg_collision = False

        # Initial position and velocity of the lander
        self.position = init_position
        self.velocity = init_velocity
        self.acceleration = Vector2D(0, 0)
        # Angle and force of the thruster
        self.angle = 0
        self.force = 0

        # disntance sensors spread out uniformly in circular pattern each one calculates the distance to the nearest surface (including map limits)
        self.num_distance_sensors = 8
        self.distance_sensors_angles = [Vector2D.remap_scalar(
            i, 0, self.num_distance_sensors, 0, 360) for i in range(self.num_distance_sensors)]
        self.distance_sensors_values = [
            -1 for _ in range(self.num_distance_sensors)]
        self.distance_sensors_collisions = [
            None for _ in range(self.num_distance_sensors)]

        self.state = np.zeros((4+self.num_distance_sensors))

    def apply_force(self, force):
        self.acceleration += force

    def add_to_surface(self, surface):
        self.surface = surface

    def step(self, desired_angle, desired_thrust, dt=1):
        # Check if the lander is out of bounds or has crashed
        if self.position.y > WORLD_HEIGHT or self.position.y < 0 or self.crashed or self.landed or self.position.x < 0 or self.position.x > WORLD_WIDTH:
            return False

        # Calculate the difference between desired angle and current angle
        angle_change = desired_angle - self.angle
        # Update the angle of the lander
        self.angle += -15 if angle_change <= - \
            15 else 15 if angle_change >= 15 else angle_change
        # Update the force of the thrusters
        self.force = desired_thrust
        # Calculate the force vector of the thruster
        thruster_force = Vector2D(0, self.force).rotate(self.angle)
        # Apply the forces of the thruster and gravity
        self.apply_force(thruster_force)
        self.apply_force(Vector2D(0, -self.GRAVITY))
        # Store the original values of velocity and position
        original_velocity = copy.copy(self.velocity)
        original_position = copy.copy(self.position)
        # Number of time to subdivide time step to check for collisions
        subdivision_tries = 2
        # Iterate over subdivisions and update the lander's position and velocity
        for i in range(1, subdivision_tries):
            self.velocity += self.acceleration * (dt/subdivision_tries)
            self.position += self.velocity * (dt/subdivision_tries)
            self.surface.collisions_check(self)
        # Restore original position and velocity and calculate displacement over dt
        self.velocity = original_velocity
        self.position = original_position
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        # Reset the acceleration to 0
        self.acceleration *= 0
        # Check one last time for collisions at the new position
        self.surface.collisions_check(self)
        if self.position.x <= self.LANDER_RADIUS or self.position.x >= WORLD_WIDTH-self.LANDER_RADIUS or\
                self.position.y <= self.LANDER_RADIUS or self.position.y >= WORLD_HEIGHT-self.LANDER_RADIUS:
            self.crashed = True
        return True

    def reset(self):
        self.__init__(INIT_POSITION, INIT_VELOCITY)

    def distance_to_landing_zone(self):
        landing_zone_segment = self.surface.segments[self.surface.landing_zone_index]
        # Calculating the closest point in the landing segment to the crash zone
        P = landing_zone_segment[0]
        Q = landing_zone_segment[1]
        X = self.position
        QP = (Q - P)
        # calculating the projection of X onto segment [P,Q]
        ds = (X - P).dot(QP) / QP.dot(QP)
        closest_point_to_landing_zone = P + ds * \
            QP if 0 < ds < 1 else P if ds <= 0 else Q
        return (self.position - closest_point_to_landing_zone).length()

    def direction_to_landing_zone(self):
        landing_zone_segment = self.surface.segments[self.surface.landing_zone_index]
        # Calculating the closest point in the landing segment to the crash zone
        P = landing_zone_segment[0]
        Q = landing_zone_segment[1]
        X = self.position
        QP = (Q - P)
        # calculating the projection of X onto segment [P,Q]
        ds = (X - P).dot(QP) / QP.dot(QP)
        closest_point_to_landing_zone = P + ds * \
            QP if 0 < ds < 1 else P if ds <= 0 else Q
        return self.position - closest_point_to_landing_zone

    def render(self, screen):
        # Calculate the position of the landing legs
        left_landing_leg = self.position - \
            Vector2D(0, self.LANDING_LEG_LENGTH).rotate(
                self.angle+self.LANDING_LEG_ANGLE)
        right_landing_leg = self.position - \
            Vector2D(0, self.LANDING_LEG_LENGTH).rotate(
                self.angle-self.LANDING_LEG_ANGLE)

        pygame.draw.line(screen,
                         (255, 0, 0) if self.left_leg_collision else (
                             255, 255, 255),
                         world_to_screen(self.position),
                         world_to_screen(left_landing_leg), 2)
        pygame.draw.line(screen,
                         (255, 0, 0) if self.right_leg_collision else (
                             255, 255, 255),
                         world_to_screen(self.position),
                         world_to_screen(right_landing_leg), 2)
        # Draw the lander body
        pygame.draw.circle(screen,
                           (255, 0, 0) if self.body_collision else (255, 255, 255),
                           world_to_screen(self.position),
                           scalar_world_to_screen(self.LANDER_RADIUS))
        for i, a in enumerate(self.distance_sensors_angles):
            pygame.draw.line(screen,
                             (255, 255, 0),
                             world_to_screen(self.position),
                             world_to_screen(self.position + Vector2D(7000, 0).rotate(a)), 2)
            if self.distance_sensors_collisions[i] is not None:
                pygame.draw.circle(screen,
                                   (0, 255, 0),
                                   world_to_screen(
                                       self.distance_sensors_collisions[i]),
                                   5)

        sensor_debug_pos = Vector2D(300, 2800)
        # Draw the sensor data in the corner of the screen
        pygame.draw.circle(screen, (150, 150, 150),
                           world_to_screen(sensor_debug_pos),
                           scalar_world_to_screen(100))
        for i, a in enumerate(self.distance_sensors_angles):
            length = min(2000, self.distance_sensors_values[i]) / 2000
            end = world_to_screen(sensor_debug_pos) +\
                Vector2D(length * scalar_world_to_screen(100), 0).rotate(-a)
            pygame.draw.line(screen, (255, 0, 255),
                             world_to_screen(sensor_debug_pos), end, 2)

        landing_debug_pos = Vector2D(500, 2800)
        landing_direction = self.direction_to_landing_zone().normalize()*100
        arrow_length = 50
        arrow_head_length = 10
        arrow_head_width = 5

        # Calculate the positions of the two points that make up the arrow head

        arrow_tail = world_to_screen(sensor_debug_pos)
        arrow_head = world_to_screen(
            sensor_debug_pos + landing_direction * arrow_length)
        left_tip = arrow_head + Vector2D(-arrow_head_width, -arrow_head_length).rotate(
            landing_direction.angle_to(Vector2D(1, 0)))
        right_tip = arrow_head + Vector2D(arrow_head_width, -arrow_head_length).rotate(
            landing_direction.angle_to(Vector2D(1, 0)))
        # Draw the arrow
        pygame.draw.line(screen, (0, 0, 255), world_to_screen(landing_debug_pos),
                         world_to_screen(landing_debug_pos + landing_direction), 2)
#        points = [arrow_tail, left_tip, right_tip]
#        pygame.draw.polygon(screen, (0, 255, 0), points, 0)

    def get_state(self):
        return self.state

    def calc_state(self):
        dir_to_land = self.direction_to_landing_zone()
        dir_info = [self.velocity.x, self.velocity.y,
                    dir_to_land.x,   dir_to_land.y]
        # Clipping the distance sensors to 1000 distance and then normalizing
        sensors = [min(2000, d) / 2000 for d in self.distance_sensors_values]
        for i, val in enumerate(dir_info + sensors):
            self.state[i] = val


class Surface:
    def __init__(self, points):
        self.points = points
        self.collisions = [False, False, False]
        self.segments = []
        # Generate the segments from the points
        for i in range(len(points) - 1):
            segment = (Vector2D(points[i]), Vector2D(points[i + 1]))
            self.segments.append(segment)
            if segment[0].y == segment[1].y:
                self.landing_zone_index = i
            self.collisions.append(False)
        self.segments.extend([(Vector2D(0, 0), Vector2D(0, WORLD_HEIGHT)),
                              (Vector2D(0, WORLD_HEIGHT), Vector2D(
                                  WORLD_WIDTH, WORLD_HEIGHT)),
                              (Vector2D(WORLD_WIDTH, 0), Vector2D(WORLD_WIDTH, WORLD_HEIGHT))])

    def collisions_check(self, lander):
        lander.right_leg_collision = False
        lander.left_leg_collision = False
        lander.body_collision = False
        lander.distance_sensors_values = [
            99999 for _ in range(lander.num_distance_sensors)]
        lander.distance_sensors_collisions = [
            None for _ in range(lander.num_distance_sensors)]
        for i, segment in enumerate(self.segments):
            self.collisions[i] = self.collides_with_lander(lander, segment)
            for j, distance_sensor_angle in enumerate(lander.distance_sensors_angles):
                ds_segment = (lander.position, lander.position +
                              Vector2D(WORLD_WIDTH, 0).rotate(distance_sensor_angle))
                hit, collision_point = self.collides_with_segment(
                    segment, ds_segment)
                temp_dist = (lander.position - collision_point).length()
                if hit and lander.distance_sensors_values[j] > temp_dist:
                    lander.distance_sensors_values[j] = temp_dist
                    lander.distance_sensors_collisions[j] = collision_point
            if not self.collisions[i]:
                continue
            if lander.angle == 0 and abs(lander.velocity.y) <= 40 and abs(lander.velocity.x) <= 20:
                if lander.left_leg_collision and lander.right_leg_collision:
                    if segment[0].y == segment[1].y:
                        lander.landed = True
                        lander.crashed = False
                        break
            lander.crashed = True
            lander.landed = False
            break

    def collides_with_lander(self, lander, segment):
        # Get the start and end points of the segment
        p1, p2 = segment

        # Calculate the distance between the lander's position and the segment
        distance = self.distance_to_segment(lander.position, p1, p2)

        rleg_segment = (lander.position, lander.position - Vector2D(
            0, lander.LANDING_LEG_LENGTH).rotate(lander.angle - lander.LANDING_LEG_ANGLE))
        lleg_segment = (lander.position, lander.position - Vector2D(
            0, lander.LANDING_LEG_LENGTH).rotate(lander.angle + lander.LANDING_LEG_ANGLE))

        rleg_collision, _ = self.collides_with_segment(segment, rleg_segment)
        lleg_collision, _ = self.collides_with_segment(segment, lleg_segment)
        # Check if the distance is less than or equal to the lander's radius
        body_collision = distance <= lander.LANDER_RADIUS
        if rleg_collision:
            lander.right_leg_collision = True

        if lleg_collision:
            lander.left_leg_collision = True

        if body_collision:
            lander.body_collision = True

        return rleg_collision or lleg_collision or body_collision

    def distance_to_segment(self, point, p1, p2):
        # Calculate the length of the segment
        length = (p1 - p2).length()
        if length == 0:
            return (point - p1).length()

        # Calculate the projection of the point onto the segment
        projection = (point - p1).dot(p2 - p1) / length
        # Check if the projection falls outside the segment
        if projection < 0:
            return (point - p1).length()
        elif projection > length:
            return (point - p2).length()

        # Calculate the distance between the point and the segment
        return (point - (p1 + projection / length * (p2 - p1))).length()

    def collides_with_segment(self, segment_1, segment_2):
        p1, p2 = segment_1
        p3, p4 = segment_2
        # Check if one of the segments is vertical
        # Check for vertical segments
        if abs(p1.x - p2.x) < .0001 and abs(p3.x - p4.x) < .0001:  # segment 1 is vertical
            if p1.x == p3.x:  # segments are coincident
                # Return any point of intersection, since the segments are coincident
                return True, p3
        elif abs(p1.x - p2.x) < .0001:
            # Segment 1 is vertical, segment 2 is not
            # Find the intersection of segment 1 with segment 2
            m2 = (p4.y - p3.y) / (p4.x - p3.x)  # slope of segment 2
            b2 = p3.y - m2 * p3.x  # y-intercept of segment 2
            y = m2 * p1.x + b2  # y-coordinate of intersection
            # Check if the intersection point is on both segments
            if min(p1.y, p2.y) <= y <= max(p1.y, p2.y) and min(p4.y, p3.y) <= y <= max(p4.y, p3.y):
                if min(p3.x, p4.x) <= p1.x <= max(p3.x, p4.x):
                    return True, Vector2D(p1.x, y)
        elif abs(p3.x - p4.x) < .0001:  # segment 2 is vertical
            # Find the intersection of segment 2 with segment 1
            m1 = (p2.y - p1.y) / (p2.x - p1.x)  # slope of segment 1
            b1 = p1.y - m1 * p1.x  # y-intercept of segment 1
            y = m1 * p3.x + b1  # y-coordinate of intersection
            # Check if the intersection point is on both segments
            if min(p1.y, p2.y) <= y <= max(p1.y, p2.y) and min(p4.y, p3.y) <= y <= max(p4.y, p3.y):
                if min(p1.x, p2.x) <= p3.x <= max(p1.x, p2.x):
                    return True, Vector2D(p3.x, y)
        else:
            # Convert the line segments to the form y = mx + b
            m1 = (p2.y - p1.y) / (p2.x - p1.x) if p2.x != p1.x else float('inf')
            b1 = p1.y - m1 * p1.x
            m2 = (p4.y - p3.y) / (p4.x - p3.x) if p4.x != p3.x else float('inf')
            b2 = p3.y - m2 * p3.x
            # Check if the segments are colliding
            if m1 != m2:
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                # Calculate the x-coordinate of the intersection point
                # Check if the x-coordinate falls within both segments
                if (x >= p1.x and x <= p2.x) or (x >= p2.x and x <= p1.x):
                    if (x >= p3.x and x <= p4.x) or (x >= p4.x and x <= p3.x):
                        return True, Vector2D(x, y)
            else:
                # Check if the y-intercepts are equal (coincident)
                if b1 == b2:
                    if (p3 - p1).length() < (p3 - p2).length():
                        return True, p2
                    else:
                        return True, p1

        return False, Vector2D(99999, 99999)

    def reset(self):
        self.collisions = [False for _ in range(len(self.segments))]

    def render(self, screen):
        i = 0
        for segment, collided in zip(self.segments, self.collisions):
            start, end = segment
            line_color = (255, 255, 255)
            # print(lander)
            if collided:
                line_color = (255, 0, 0)
            elif i == self.landing_zone_index:
                line_color = (0, 255, 0)
            pygame.draw.line(screen, line_color,
                             world_to_screen(start),
                             world_to_screen(end), 2)
            i += 1


def world_to_screen(point):
    pointx = Vector2D.remap_scalar(point.x, 0, WORLD_WIDTH,
                                   0, SCREEN_WIDTH)
    pointy = Vector2D.remap_scalar(point.y, 0, WORLD_HEIGHT,
                                   0, SCREEN_HEIGHT)
    pointy = SCREEN_HEIGHT - pointy
    return Vector2D(pointx, pointy)


def scalar_world_to_screen(n):
    return Vector2D.remap_scalar(n, 0, WORLD_HEIGHT,
                                 0, SCREEN_HEIGHT)


class MarsLanderEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, init_position, init_velocity, surface_topology, render_mode='rgb_array'):
        self.lander = Lander(init_position, init_velocity)
        self.surface = Surface(surface_topology)
        self.lander.add_to_surface(self.surface)
        #low_pos = np.array([0, 0], dtype=np.float32)
        #high_pos = np.array([7000, 3000], dtype=np.float32)
        low_vel = np.array([-1000, -1000], dtype=np.float32)
        high_vel = np.array([1000, 1000], dtype=np.float32)
#        low_angle = np.array([-90], dtype=np.float32)
#        high_angle = np.array([90], dtype=np.float32)
#        low_thrust = np.array([2], dtype=np.float32)
#        high_thrust = np.array([4], dtype=np.float32)
        low_dir = np.array([-4000, -4000], dtype=np.float32)
        high_dir = np.array([4000, 4000], dtype=np.float32)
        low_sensors = np.zeros(
            self.lander.num_distance_sensors, dtype=np.float32)
        high_sensors = np.full(
            self.lander.num_distance_sensors, 1, dtype=np.float32)
        low = np.concatenate(
            (low_vel, low_dir, low_sensors))
        high = np.concatenate(
            (high_vel, high_dir, high_sensors))
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array(
            [3, -6]), high=np.array([4, 6]), dtype=np.int32)

        self.screen = None
        self.render_mode = render_mode
        self.clock = None

    def reset(self):
        self.lander.reset()
        self.surface.reset()
        self.lander.add_to_surface(self.surface)

        if self.render_mode == "human":
            self.render()
        info = {}
        return self.lander.get_state(), info

    def step(self, action):
        thrust = action // (6 + 6 + 1) + 3
        angle = (action % (6 + 6 + 1) - 6) * 15
        action = [thrust, angle]
        self.lander.step(action[1], action[0])

        state = self.lander.get_state()
        assert len(state) == 4 + self.lander.num_distance_sensors

        reward = 2000
        landing_zone_segment = self.surface.segments[self.surface.landing_zone_index]
        landing_zone_midpoint = (
            landing_zone_segment[0] + landing_zone_segment[1])/2
        distance_from_landing_zone = (
            self.lander.position - landing_zone_midpoint).length()
        reward -= distance_from_landing_zone / 2
        reward -= abs(self.lander.velocity.x)
        reward -= abs(self.lander.velocity.y)/2

        if self.lander.landed:
            reward = 3000
        done = self.lander.landed or self.lander.crashed
        info = {}

        if self.render_mode == "human":
            self.render()
        return state, reward, done, info

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.lander.render(self.surf)
        self.surface.render(self.surf)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


# Set up the Pygame window
WORLD_WIDTH, WORLD_HEIGHT = 7000, 3000
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 700
# Create a lander
SURFACE_TOPOLOGY = [(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600),
                    (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)]
INIT_POSITION = Vector2D(6500, 2600)
INIT_VELOCITY = Vector2D(0, 0)

env = MarsLanderEnv(
    INIT_POSITION, INIT_VELOCITY, SURFACE_TOPOLOGY, render_mode='human')
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = 2 * 13
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(50000)

episode_durations = []

steps_done = 0
"""
# Load the latest episode number and steps_done
latest_episode = max([int(filename.split("_")[2][2:]) for filename in os.listdir(
    "models/.99g_1e-4_50000_2000_rand_startx10_50k") if filename.startswith("policy_net")])
latest_steps = max([int(filename.split("_")[3][:-4]) for filename in os.listdir(
    "models/.99g_1e-4_50000_2000_rand_startx10_50k") if filename.startswith(f"policy_net_ep{latest_episode}")])

# Load the model state dictionaries
print(
    f"Loading 'models/.99g_1e-4_50000_2000_rand_startx10_50k/policy_net_ep{latest_episode}_{latest_steps}.pth' ...")
policy_net.load_state_dict(torch.load(
    f"models/.99g_1e-4_50000_2000_rand_startx10_50k/policy_net_ep{latest_episode}_{latest_steps}.pth"))
print(
    f"Loading 'models/.99g_1e-4_50000_2000_rand_startx10_50k/target_net_ep{latest_episode}_{latest_steps}.pth' ...")
target_net.load_state_dict(torch.load(
    f"models/.99g_1e-4_50000_2000_rand_startx10_50k/target_net_ep{latest_episode}_{latest_steps}.pth"))

print(
    f"Loading 'models/.99g_1e-4_50000_2000_rand_startx10_50k/memory_buffer_ep{latest_episode}_{latest_steps}.pkl' ...")

# Load the memory buffer and episode rewards
with open(f"models/.99g_1e-4_50000_2000_rand_startx10_50k/memory_buffer_ep{latest_episode}_{latest_steps}.pkl", "rb") as f:
    memory = pkl.load(f)
print(
    f"Loading 'models/.99g_1e-4_50000_2000_rand_startx10_50k/episode_rewards_ep{latest_episode}_{latest_steps}.pkl' ...")

with open(f"models/.99g_1e-4_50000_2000_rand_startx10_50k/episode_rewards_ep{latest_episode}_{latest_steps}.pkl", "rb") as f:
    episode_durations = pkl.load(f)
steps_done = latest_steps
"""


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        t, a = env.action_space.sample()
        encoded_action = (t - 3) * (6 + 6 + 1) + a + 6
        return torch.tensor(np.array([[encoded_action]]), device=device, dtype=torch.long)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 6000
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
    if i_episode % 1000 == 0:
        with open(f"models/memory_buffer_ep{i_episode:}_{steps_done}.pkl", 'wb') as f:
            pkl.dump(memory, f)
        with open(f"models/episode_rewards_ep{i_episode:}_{steps_done}.pkl", "wb") as f:
            pkl.dump(episode_durations, f)

        torch.save(policy_net.state_dict(),
                   f"models/policy_net_ep{i_episode:}_{steps_done}.pth")
        torch.save(target_net.state_dict(),
                   f"models/target_net_ep{i_episode:}_{steps_done}.pth")
    if i_episode % 1 == 0:
        INIT_POSITION.x = random.randint(200, WORLD_WIDTH-200)
    #steps_done = 0
    for t in count():
        # Debug message
        action = select_action(state)
        observation, reward, terminated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated
        if t % 50 == 0 or done:
            print(
                f"Step {t} of episode {i_episode} {env.lander.position} {reward.item()}")
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(reward)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
