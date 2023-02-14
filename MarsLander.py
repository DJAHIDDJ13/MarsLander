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
        self.position = np.array(self.INITIAL_POSITION)
        self.velocity = np.array(self.INITIAL_VELOCITY)
        self.acceleration = np.array([0, 0])
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
        for i, a in enumerate(self.distance_sensors_angles):
            pygame.draw.line(screen,
                             (255, 255, 0),
                             world_to_screen(self.position),
                             world_to_screen(self.position + Lander.rotate_vec(np.array([7000, 0]), a)), 2)
            if self.distance_sensors_collisions[i] is not None:
                pygame.draw.circle(screen,
                                   (0, 255, 0),
                                   world_to_screen(
                                       self.distance_sensors_collisions[i]),
                                   5)

    def get_state(self):
        return self.state

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
        dir_info = [self.velocity[0], self.velocity[1],
                    dir_to_land[0],   dir_to_land[1]]
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
            [3, -6]), high=np.array([4, 6]), dtype=np.int32)

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
        self.lander.step(*action)

        state = self.lander.get_state()
#        assert len(state) == 4 + self.lander.num_distance_sensors

        reward = 0
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


# Create a lander
SURFACE_TOPOLOGY = [(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600),
                    (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)]
SURFACE_TOPOLOGY = [(0, 1800), (300, 1200), (1000, 1550), (2000, 1200), (2500, 1650), (3700, 220), (4700, 220), (4750, 1000), (4700, 1650),
                    (4000, 1700), (3700, 1600), (3750, 1900), (4000, 2100), (4900, 2050), (5100, 1000), (5500, 500), (6200, 800), (6999, 600)]
INIT_POSITION = np.array([6500, 2000])
INIT_VELOCITY = np.array([0, 0])
lander_env = MarsLanderEnv(
    INIT_POSITION, INIT_VELOCITY, SURFACE_TOPOLOGY, render_mode='human')
lander_env.render()
desired_angle, desired_thrust = 0, 0
i = 0
random.seed(1)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pass
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                desired_angle = 10
            elif event.key == pygame.K_RIGHT:
                desired_angle = -10
            elif event.key == pygame.K_UP:
                desired_thrust = 4
            elif event.key == pygame.K_DOWN:
                desired_thrust = 0
            elif event.key == pygame.K_F5:
                # Reset the lander and surface
                lander_env.reset()
        elif event.type == pygame.KEYUP:
            # Reset the desired angle and thrust when the keys are released
            desired_angle = 0
            desired_thrust = 0

    action = [desired_thrust, desired_angle]
    observation, reward, done, info = lander_env.step(action)
    i += 1
    print(i, lander_env.lander.position.round(),
          lander_env.lander.angle, lander_env.lander.thrust, lander_env.lander.velocity.round(), lander_env.lander.landed)
    #print(observation, reward, done, info)
    if done:
        print(lander_env.lander.landed)
        break
lander_env.close()
