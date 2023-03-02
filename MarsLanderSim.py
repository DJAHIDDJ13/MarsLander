import pygame
import numpy as np
import gym
import time

# Set up the Pygame window
WORLD_WIDTH, WORLD_HEIGHT = 7000, 3000
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 700


def remap_scalar(x, old_min, old_max, new_min, new_max):
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


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


class Lander:
    EPS = .0000001
    WORLD_WIDTH = 7000
    WORLD_HEIGHT = 3000

    def __init__(self, init_position, init_velocity, ground, calc_sensors=True):
        # Constants
        self.GRAVITY = 3.711
        self.LANDER_RADIUS = 50
        self.LANDING_LEG_ANGLE = 25
        # from the center of the body
        self.LANDING_LEG_LENGTH = 100

        self.crashed = False
        self.landed = False
        self.calc_sensors = calc_sensors

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

        # too much stuff here, TODO: sort it out and remove what isn't needed
        num_ground_points = len(ground) + 2

        self.landing_zones = []
        self.landing_zone_indices = []
        lasty = -1
        lastx = -1
        self.collision_poly = np.zeros((num_ground_points, 2))
        for i, point in enumerate(ground):
            self.collision_poly[i][0] = point[0]
            self.collision_poly[i][1] = point[1]
            if point[1] == lasty:
                self.landing_zones.append(
                    [min(lastx, point[0]), max(lastx, point[0]), point[1]])
                self.landing_zone_indices.append(i-1)
            lastx, lasty = point
        self.collision_poly[-2][0] = self.WORLD_WIDTH
        self.segments = [*map(np.array, zip(ground, ground[1:]))]
        self.segments.extend([np.array([ground[-1],
                                        [Lander.WORLD_WIDTH, Lander.WORLD_HEIGHT]]),

                              np.array([[Lander.WORLD_WIDTH, Lander.WORLD_HEIGHT],
                                        [0, Lander.WORLD_HEIGHT]]),

                              np.array([[0, Lander.WORLD_HEIGHT],
                                        ground[0]])])
        self.landing_zones = np.array(self.landing_zones)

        polygon = np.vstack((np.array(ground), [7000, 3000], [0, 3000])).T
        points_aa = np.tile(polygon, (8, 1, 1))
        points_ab = np.roll(points_aa, -1, axis=2)
        points_aa = points_aa.transpose((1, 0, 2))
        points_ab = points_ab.transpose((1, 0, 2))
        self.poly_segments = (points_aa, points_ab)
        self.poly_segments_no_sensors = (
            polygon, np.roll(polygon.T, -1, axis=0).T)
        # disntance sensors spread out uniformly in circular pattern each one
        # calculates the distance to the nearest surface (including map limits)
        self.num_distance_sensors = 8
        self.distance_sensors_angles = np.array([remap_scalar(
            i, 0, self.num_distance_sensors, 0, 360) for i in range(self.num_distance_sensors)])
        rad = self.distance_sensors_angles * np.pi / 180
        ray_length = 9000
        self.sensor_rays = np.column_stack(
            (ray_length * np.cos(rad), ray_length * np.sin(rad)))
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

    def collision_check(self):
        points_aa, points_ab = self.poly_segments_no_sensors
        points_ba = np.tile(self.position, (len(points_aa[0]), 1))
        # setting the segment to just right of the map
        points_bb = np.tile(np.array([7001, self.position[1]]),
                            (len(points_aa[0]), 1))

        x1, y1 = points_aa
        x2, y2 = points_ab
        dx1, dy1 = x2 - x1, y2 - y1

        x3, y3 = points_ba.T
        x4, y4 = points_bb.T
        dx2, dy2 = x4 - x3, y4 - y3

        # Calculate the determiants
        det = dx1*dy2 - dy1*dx2

        # Calculate the parameter values for the intersection point
        # div by zero retuns -inf and raises warning, no problems in calculation however
        # Might be worth it to supress the warning temporarily here,
        # using a np.where instead seems overkill
        t1 = (dy2*(x3-x1) + dx2*(y1-y3)) / det
        t2 = (dy1*(x3-x1) + dx1*(y1-y3)) / det

        collisions = (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)

        return collisions.sum() % 2 == 0

    # doing both at once to optimize

    def collision_check_with_sensors(self):
        points_aa, points_ab = self.poly_segments
        points_ba = np.tile(
            self.position[np.newaxis, :].T, (1, len(points_aa[0])))
        points_ba = np.tile(
            self.position[np.newaxis, :, np.newaxis], (8, 1, points_aa.shape[2]))
        points_bb = self.sensor_rays + self.position[np.newaxis, :]
        points_bb = np.tile(
            points_bb[:, :, np.newaxis], (1, 1, points_aa.shape[2]))

        # change the previous lines to avoid this
        points_ba = points_ba.transpose((1, 0, 2))
        points_bb = points_bb.transpose((1, 0, 2))
        # setting the segment to just right of the map
        x1, y1 = points_aa
        x2, y2 = points_ab
        dx1, dy1 = x2 - x1, y2 - y1

        x3, y3 = points_ba
        x4, y4 = points_bb
        dx2, dy2 = x4 - x3, y4 - y3

        # Calculate the determiants
        det = dx1*dy2 - dy1*dx2

        # Calculate the parameter values for the intersection point
        # div by zero retuns -inf and raises warning, no problems in calculation however
        # Might be worth it to supress the warning temporarily here,
        # using a np.where instead seems overkill
        x3_x1 = x3-x1
        y1_y3 = y1-y3
        t1 = (dy2 * x3_x1 + dx2 * y1_y3) / det
        t2 = (dy1 * x3_x1 + dx1 * y1_y3) / det

        collisions = (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)
        self.distance_sensors_collisions = [self.position + t2[i, collisions[i, :]][0] * self.sensor_rays[i]
                                            if len(t2[i, collisions[i, :]]) else None for i in range(len(t2))]
        self.distance_sensors_values = [t2[i, collisions[i, :]][0] * np.linalg.norm(self.sensor_rays[i])
                                        if len(t2[i, collisions[i, :]]) else 99999 for i in range(len(t2))]
        # any row should work to figure out if the lander is inside the polygon
        return collisions[0, :].sum() % 2 == 0

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
        col = self.collision_check_with_sensors(
        ) if self.calc_sensors else self.collision_check()
        if col:
            if self.angle == 0 and (abs(self.velocity) <= (20, 40)).all():
                for segment in self.landing_zones:
                    if segment[0] <= self.position[0] <= segment[1] and abs(self.position[1] - segment[2]) < 100:
                        self.landed = True
                        return
            self.crashed = True

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
        """
        sensor_debug_pos = np.array([300, 2800], dtype="float64")

        for i, a in enumerate(self.distance_sensors_angles):
            pygame.draw.line(screen,
                             (255, 255, 0),
                             world_to_screen(sensor_debug_pos),
                             world_to_screen(sensor_debug_pos +
                                             Lander.rotate_vec(self.state[4+i]*np.array([100, 0]), a)),
                             1)
        # d = self.direction_to_landing_zone()
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
        """

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

    def calc_state(self):
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


class MarsLanderEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    SCREEN_WIDTH = 1400
    SCREEN_HEIGHT = 700

    def __init__(self, init_position, init_velocity, surface_topology, calc_sensors=True, render_mode='rgb_array'):
        self.lander = Lander(init_position, init_velocity,
                             surface_topology, calc_sensors)

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
        if isinstance(action, int):
            thrust = action // (6 + 6 + 1)
            angle = (action % (6 + 6 + 1) - 6) * 15
            action = [thrust, angle]

        self.lander.step(action[0], action[1])

        state = self.lander.state
        assert len(state) == 4 + self.lander.num_distance_sensors

        reward = 1
        distance_from_landing_zone = self.lander.distance_to_landing_zone()
        reward -= distance_from_landing_zone / 4000
        reward -= abs(self.lander.velocity[0]) / 300
        reward -= abs(self.lander.velocity[1]) / 150
        # reset the reward if it's not the final step, we only reward at the end
        if not self.lander.crashed:
            reward = 0
        if self.lander.landed:
            reward = 10

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


def demo_manual_lander():
    print("Running manual controlled lander")
    SURFACE_TOPOLOGY = [(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600),
                        (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)]
    INIT_POSITION = np.array([6500, 2600])
    INIT_VELOCITY = np.array([0, 0])
    env = MarsLanderEnv(
        INIT_POSITION, INIT_VELOCITY, SURFACE_TOPOLOGY, calc_sensors=True, render_mode='human')
    desired_thrust, desired_angle = 0, 0
    s, info = env.reset()
    i = 0
    while True:
        time.sleep(.05)
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
                    env.reset()
            elif event.type == pygame.KEYUP:
                # Reset the desired angle and thrust when the keys are released
                desired_angle = 0
                desired_thrust = 0
        action = [desired_thrust, desired_angle]
        observation, reward, done, info = env.step(action)
        i += 1
        print(i, env.lander.position.round(),
              env.lander.angle, env.lander.thrust,
              env.lander.velocity.round(), env.lander.landed)
        # print(observation, reward, done, info)
        if done:
            print("Landed status:", env.lander.landed)
            break

    env.close()


if __name__ == "__main__":
    demo_manual_lander()
