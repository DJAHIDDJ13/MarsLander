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


def remap_scalar(x, old_min, old_max, new_min, new_max):
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class Lander:
    EPS = .0000001
    WORLD_WIDTH = 7000
    WORLD_HEIGHT = 3000

    def __init__(self, init_position, init_velocity, ground):
        # Constants
        self.GRAVITY = 3.711
        self.LANDER_RADIUS = 50
        self.LANDING_LEG_ANGLE = 25
        # from the center of the body
        self.LANDING_LEG_LENGTH = 100

        self.crashed = False
        self.landed = False

        # Initial position and velocity of the lander
        self.INITIAL_POSITION = init_position
        self.INITIAL_VELOCITY = init_velocity
        self.position = np.array(init_position, dtype="float64")
        self.velocity = np.array(init_velocity, dtype="float64")
        self.acceleration = np.array([0, 0], dtype="float64")
        # Angle and force of the thruster
        self.angle = 0
        self.thrust = 0
        self.fuel = 0
        self.turn = 0

        num_ground_points = len(ground) + 2

        self.landing_zones = []
        self.collision_poly = np.zeros((num_ground_points, 2))
        self.landing_zone_indices = []
        lasty = -1
        lastx = -1
        for i, point in enumerate(ground):
            self.collision_poly[i][0] = point[0]
            self.collision_poly[i][1] = point[1]
            if point[1] == lasty:
                self.landing_zones.append(
                    [min(lastx, point[0]), max(lastx, point[0]), point[1]])
                self.landing_zone_indices.append(i-1)
            lastx, lasty = point
        self.collision_poly[-2][0] = self.WORLD_WIDTH

        self.landing_zones = np.array(self.landing_zones)
        self.segments = [*map(np.array, zip(ground, ground[1:]))]
        self.segments.extend([np.array([ground[-1],
                                        [Lander.WORLD_WIDTH, Lander.WORLD_HEIGHT]]),

                              np.array([[Lander.WORLD_WIDTH, Lander.WORLD_HEIGHT],
                                        [0, Lander.WORLD_HEIGHT]]),

                              np.array([[0, Lander.WORLD_HEIGHT],
                                        ground[0]])])

        # disntance sensors spread out uniformly in circular pattern each one calculates the distance to the nearest surface (including map limits)
        self.num_distance_sensors = 8
        self.distance_sensors_angles = [remap_scalar(
            i, 0, self.num_distance_sensors, 0, 360) for i in range(self.num_distance_sensors)]
        self.distance_sensors_values = [
            -1 for _ in range(self.num_distance_sensors)]
        self.distance_sensors_collisions = [
            None for _ in range(self.num_distance_sensors)]

        self.state = np.zeros((4 + self.num_distance_sensors))

    def reset(self):
        self.position = np.array(self.INITIAL_POSITION, dtype="float64")
        self.velocity = np.array(self.INITIAL_VELOCITY, dtype="float64")
        self.acceleration = np.array([0, 0], dtype="float64")
        self.angle = 0
        self.force = 0
        self.fuel = 0
        self.landed = False
        self.crashed = False
        self.calc_state()

    def collisionCheck(self):
        angleSum = 0
        for i in range(len(self.collision_poly)):
            pointA = self.collision_poly[i]
            pointB = self.position
            pointC = self.collision_poly[(i + 1) % len(self.collision_poly)]
            if Lander.CCW(pointA, pointB, pointC):
                angleSum += Lander.angle_between(pointA, pointB, pointC)
            else:
                angleSum -= Lander.angle_between(pointA, pointB, pointC)
        return abs(abs(angleSum) - 2 * math.pi) < self.EPS

    def step(self, thrust, angle, dt=1):
        self.angle = max(
            min(max(self.angle - 15, min(self.angle + 15, angle)), 90), -90)
        self.thrust = max(self.thrust - 1, min(self.thrust + 1, thrust))
        self.fuel = max(self.fuel - self.thrust, 0)  # update fuel

        # apply the thrust force
        radians = np.radians(-self.angle)
        self.acceleration = self.thrust * \
            np.array([np.sin(radians), np.cos(radians)])
        # apply gravity force
        self.acceleration[1] -= self.GRAVITY

        # Update the position and velocity
        if not self.crashed and not self.landed:
            self.position += self.velocity * dt + self.acceleration * dt / 2
            self.velocity += self.acceleration * dt

        self.turn += 1
        self.calc_state()
        if self.position[0] < 0 or self.position[0] >= self.WORLD_WIDTH or \
           self.position[1] < 0 or self.position[1] >= self.WORLD_HEIGHT:
            self.crashed = True
            return
        if self.collisionCheck():
            if self.angle == 0 and (abs(self.velocity) <= (20, 40)).all():
                for segment in self.landing_zones:
                    if segment[0] <= self.position[0] <= segment[1] and abs(self.position[1] - segment[2]) < 100:
                        self.landed = True
                        return
            self.crashed = True

    @staticmethod
    def angle_between(a, b, c):
        ba = b - a
        bc = b - c
        dot_product = np.dot(ba, bc)
        ba_magnitude = np.linalg.norm(ba)
        bc_magnitude = np.linalg.norm(bc)
        return np.arccos(dot_product / (ba_magnitude * bc_magnitude))

    @staticmethod
    def CCW(a, b, c):
        return np.cross(b - a, b - c) > -Lander.EPS

    @staticmethod
    def rotate_vec(v, angle):
        radians = np.radians(angle)
        cos_val = np.cos(radians)
        sin_val = np.sin(radians)
        return np.array([v[0]*cos_val - v[1]*sin_val, v[0]*sin_val + v[1]*cos_val])

    def render(self, screen):
        # Calculate the position of the landing legs
        radians = np.radians(self.angle)
        angle_left = radians + 90 + self.LANDING_LEG_ANGLE
        angle_right = radians + 90 - self.LANDING_LEG_ANGLE
        left_landing_leg = self.position - \
            Lander.rotate_vec(
                np.array([self.LANDING_LEG_LENGTH, 0]),
                angle_left)
        right_landing_leg = self.position - \
            Lander.rotate_vec(
                np.array([self.LANDING_LEG_LENGTH, 0]),
                angle_right)

        pygame.draw.line(screen,
                         (255, 255, 255),
                         world_to_screen(self.position),
                         world_to_screen(left_landing_leg), 2)
        pygame.draw.line(screen,
                         (255, 255, 255),
                         world_to_screen(self.position),
                         world_to_screen(right_landing_leg), 2)
        # Draw the lander body
        pygame.draw.circle(screen,
                           (255, 255, 255),
                           world_to_screen(self.position),
                           scalar_world_to_screen(self.LANDER_RADIUS))
        pygame.draw.polygon(screen, (255, 255, 255),
                            [*map(world_to_screen, self.collision_poly)], 0)

        for i in self.landing_zone_indices:
            start, end = self.segments[i]
            pygame.draw.line(screen, (0, 255, 0),
                             world_to_screen(start),
                             world_to_screen(end), 2)
        """for i, a in enumerate(self.distance_sensors_angles):
            pygame.draw.line(screen,
                             (255, 255, 0),
                             world_to_screen(self.position),
                             world_to_screen(self.position + Lander.rotate_vec(np.array([7000, 0]), a)), 2)
            if self.distance_sensors_collisions[i] is not None:
                pygame.draw.circle(screen,
                                   (0, 255, 0),
                                   world_to_screen(
                                       self.distance_sensors_collisions[i]),
                                   5)"""
        sensor_debug_pos = np.array([300, 2800], dtype="float64")

        for i, a in enumerate(self.distance_sensors_angles):
            pygame.draw.line(screen,
                             (255, 255, 0),
                             world_to_screen(sensor_debug_pos),
                             world_to_screen(sensor_debug_pos +
                                             Lander.rotate_vec(self.state[4+i]*np.array([100, 0]), a)),
                             1)
        #d = self.direction_to_landing_zone()
        dir_debug_pos = np.array([500, 2800], dtype="float64")
        # Convert the polar coordinates to Cartesian coordinates
        p = np.array([self.state[2], self.state[3]])
        d = np.array([p[0] * np.cos(p[1]), p[0] * np.sin(p[1])])
        d /= np.linalg.norm(d)
        pygame.draw.line(screen,
                         (255, 255, 0),
                         world_to_screen(dir_debug_pos),
                         world_to_screen(dir_debug_pos + 100*d),
                         1)
        # Create a font object
        font = pygame.font.SysFont('Monospace', 15)

        # Render each value in the self.state array as a text surface
        names = ["vel_x", "vel_y", "dir_r", "dir_a",
                 "sen_1", "sen_2", "sen_3", "sen_4", "sen_5", "sen_6", "sen_7", "sen_8"]
        for i, value in enumerate(self.state):
            text = font.render(
                f"{names[i]} = {value:4.2f}", False, (255, 255, 255))
            screen.blit(text, world_to_screen([100, 2700 - i*50]))

    def get_state(self):
        self.calc_state()  # idk there is a problem somewhere,
        # why am i even doing it like this
        return self.state

    def distance_to_landing_zone(self):
        return np.linalg.norm(self.direction_to_landing_zone())

    def direction_to_landing_zone(self):
        # TODO make this for the closest landing zone; Not necessary probably
        landing_zone_segment = self.segments[self.landing_zone_indices[0]]
        # Calculating the closest point in the landing segment to the crash zone
        P = landing_zone_segment[0]
        Q = landing_zone_segment[1]
        X = self.position
        QP = (Q - P)
        # calculating the projection of X onto segment [P,Q]
        ds = (X - P).dot(QP) / QP.dot(QP)
        closest_point_to_landing_zone = P + ds * \
            QP if 0 < ds < 1 else P if ds <= 0 else Q
        return closest_point_to_landing_zone - self.position

    @staticmethod
    def segment_collision(segment_1, segment_2):
        p1, p2 = segment_1
        p3, p4 = segment_2
        # Check if one of the segments is vertical
        # Check for vertical segments
        if abs(p1[0] - p2[0]) < .0001 and abs(p3[0] - p4[0]) < .0001:  # segment 1 is vertical
            if p1[0] == p3[0]:  # segments are coincident
                # Return any point of intersection, since the segments are coincident
                return True, p3
        elif abs(p1[0] - p2[0]) < .0001:
            # Segment 1 is vertical, segment 2 is not
            # Find the intersection of segment 1 with segment 2
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])  # slope of segment 2
            b2 = p3[1] - m2 * p3[0]  # y-intercept of segment 2
            y = m2 * p1[0] + b2  # y-coordinate of intersection
            # Check if the intersection point is on both segments
            if min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and min(p4[1], p3[1]) <= y <= max(p4[1], p3[1]):
                if min(p3[0], p4[0]) <= p1[0] <= max(p3[0], p4[0]):
                    return True, np.array([p1[0], y])
        elif abs(p3[0] - p4[0]) < .0001:  # segment 2 is vertical
            # Find the intersection of segment 2 with segment 1
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])  # slope of segment 1
            b1 = p1[1] - m1 * p1[0]  # y-intercept of segment 1
            y = m1 * p3[0] + b1  # y-coordinate of intersection
            # Check if the intersection point is on both segments
            if min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and min(p4[1], p3[1]) <= y <= max(p4[1], p3[1]):
                if min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]):
                    return True, np.array([p3[0], y])
        else:
            # Convert the line segments to the form y = mx + b
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0]
                                    ) if p2[0] != p1[0] else float('inf')
            b1 = p1[1] - m1 * p1[0]
            m2 = (p4[1] - p3[1]) / (p4[0] - p3[0]
                                    ) if p4[0] != p3[0] else float('inf')
            b2 = p3[1] - m2 * p3[0]
            # Check if the segments are colliding
            if m1 != m2:
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                # Calculate the x-coordinate of the intersection point
                # Check if the x-coordinate falls within both segments
                if (x >= p1[0] and x <= p2[0]) or (x >= p2[0] and x <= p1[0]):
                    if (x >= p3[0] and x <= p4[0]) or (x >= p4[0] and x <= p3[0]):
                        return True, np.array([x, y])
            else:
                # Check if the y-intercepts are equal (coincident)
                if b1 == b2:
                    if np.linalg.norm(p3 - p1) < np.linalg.norm(p3 - p2):
                        return True, p2
                    else:
                        return True, p1

        return False, np.array([99999, 99999])

    def calc_distance_sensors(self):
        for j, distance_sensor_angle in enumerate(self.distance_sensors_angles):
            self.distance_sensors_values[j] = 9999999
            for segment in self.segments:
                ds_segment = (self.position, self.position +
                              self.rotate_vec(np.array([self.WORLD_WIDTH, 0]), distance_sensor_angle))  # rotation is wrong
                hit, collision_point = Lander.segment_collision(
                    segment, ds_segment)
                # print(f"{collision_point=}")
                temp_dist = np.linalg.norm(self.position - collision_point)
                if hit and self.distance_sensors_values[j] > temp_dist:
                    self.distance_sensors_values[j] = temp_dist
                    self.distance_sensors_collisions[j] = collision_point

    def calc_state(self):
        self.calc_distance_sensors()
        dir_to_land = self.direction_to_landing_zone()
        # Convert dir_to_land to polar coordiantes
        r = np.hypot(dir_to_land[0], dir_to_land[1])
        theta = np.arctan2(dir_to_land[1], dir_to_land[0])

        dir_info = [self.velocity[0], self.velocity[1],
                    r,   theta]
        # Clipping the distance sensors to 1000 distance and then normalizing
        sensors = [min(2000, d) / 2000 for d in self.distance_sensors_values]
        for i, val in enumerate(dir_info + sensors):
            self.state[i] = val


def world_to_screen(point):
    pointx = remap_scalar(point[0], 0, Lander.WORLD_WIDTH,
                          0, MarsLanderEnv.SCREEN_WIDTH)
    pointy = remap_scalar(point[1], 0, Lander.WORLD_HEIGHT,
                          0, MarsLanderEnv.SCREEN_HEIGHT)
    pointy = MarsLanderEnv.SCREEN_HEIGHT - pointy
    return np.array([pointx, pointy])


def scalar_world_to_screen(n):
    return remap_scalar(n, 0, Lander.WORLD_HEIGHT,
                        0, MarsLanderEnv.SCREEN_HEIGHT)


class MarsLanderEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    SCREEN_WIDTH = 1400
    SCREEN_HEIGHT = 700

    def __init__(self, init_position, init_velocity, surface_topology, render_mode='rgb_array'):
        self.lander = Lander(init_position, init_velocity, surface_topology)

        low_vel = np.array([-1000, -1000], dtype=np.float32)
        high_vel = np.array([1000, 1000], dtype=np.float32)

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
            [0, -6]), high=np.array([4, 6]), dtype=np.int32)

        self.screen = None
        self.render_mode = render_mode
        self.clock = None

    def reset(self):
        self.lander.reset()

        if self.render_mode == "human":
            self.render()
        info = {}
        return self.lander.get_state(), info

    def step(self, action):
        thrust = action // (6 + 6 + 1)
        angle = (action % (6 + 6 + 1) - 6) * 15
        action = [thrust, angle]

        self.lander.step(action[1], action[0])

        state = self.lander.get_state()
        assert len(state) == 4 + self.lander.num_distance_sensors

        reward = 2000

        distance_from_landing_zone = self.lander.distance_to_landing_zone()
        reward -= distance_from_landing_zone / 2
        reward -= abs(self.lander.velocity[0])
        reward -= abs(self.lander.velocity[1])/2
        # reset the reward if it's not the final step, we only reward at the end
#        if not self.lander.crashed:
#            reward = 0
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
                (MarsLanderEnv.SCREEN_WIDTH, MarsLanderEnv.SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (MarsLanderEnv.SCREEN_WIDTH, MarsLanderEnv.SCREEN_HEIGHT))
        self.lander.render(self.surf)

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
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Set up the Pygame window
WORLD_WIDTH, WORLD_HEIGHT = 7000, 3000
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 700
SURFACE_TOPOLOGY = [(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600),
                    (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)]
INIT_POSITION = np.array([6500, 2600])
INIT_VELOCITY = np.array([0, 0])
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
n_actions = 5 * 13
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

episode_durations = []

steps_done = 0


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
        encoded_action = t * (6 + 6 + 1) + a + 6
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
        INIT_POSITION[0] = random.randint(200, WORLD_WIDTH-200)
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
