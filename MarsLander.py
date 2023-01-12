import math
import pygame
import time
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
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
    def __init__(self, x, y, vx, vy):
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
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(vx, vy)
        self.acceleration = Vector2D(0, 0)
        # Angle and force of the thruster
        self.angle = 0
        self.force = 0

        # disntance sensors spread out uniformly in circular pattern each one calculates the distance to the nearest surface (including map limits)
        self.num_distance_sensors = 8
        self.distance_sensors_angles = [Vector2D.remap_scalar(i, 0, self.num_distance_sensors, 0, 360) for i in range(self.num_distance_sensors)]
        self.distance_sensors_values = [-1 for _ in range(self.num_distance_sensors)]
        self.distance_sensors_collisions = [None for _ in range(self.num_distance_sensors)]

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
        self.angle += -15 if angle_change <= -15 else 15 if angle_change >= 15 else angle_change
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
        return True

    def reset(self):
            self.__init__(START_X, START_Y, 0, 0)

    def run(self, chromosome):
        path = []
        for gene in chromosome.genes:
            dist = self.distance_to_landing_zone()
            if dist < 100:            
                self.step(0, gene.thrust, 1)
            else:
                self.step(gene.angle, gene.thrust, 1)
            path.append([lander.position.x, lander.position.y, lander.velocity.x, lander.velocity.y])
            if self.crashed or self.landed:
                break
        return path
    
    def get_state(self):
        dir_to_land = self.direction_to_landing_zone()
        return np.append(np.array([self.position.x, self.position.y,
                        self.velocity.x, self.velocity.y,
                        dir_to_land.x,   dir_to_land.y,
                        self.angle, self.force]),
                         self.distance_sensors_values)
                        
    def run_(self, nn_chromosome):
        path = []
        while True:
            thrust, angle = nn_chromosome.forward(self.get_state())
            # Step the simulation
            self.step(angle, thrust, 1)
            if self.crashed or self.landed:
                break
        return path        

    def distance_to_landing_zone(self):
        landing_zone_segment = self.surface.segments[lander.surface.landing_zone_index]
        # Calculating the closest point in the landing segment to the crash zone
        P = landing_zone_segment[0]
        Q = landing_zone_segment[1]
        X = lander.position
        QP = (Q - P)
        ds = (X - P).dot(QP) / QP.dot(QP) # calculating the projection of X onto segment [P,Q]
        closest_point_to_landing_zone = P + ds * QP if 0 < ds < 1 else P if ds <= 0 else Q
        return (self.position - closest_point_to_landing_zone).length()

    def direction_to_landing_zone(self):
        landing_zone_segment = self.surface.segments[lander.surface.landing_zone_index]
        # Calculating the closest point in the landing segment to the crash zone
        P = landing_zone_segment[0]
        Q = landing_zone_segment[1]
        X = lander.position
        QP = (Q - P)
        ds = (X - P).dot(QP) / QP.dot(QP) # calculating the projection of X onto segment [P,Q]
        closest_point_to_landing_zone = P + ds * QP if 0 < ds < 1 else P if ds <= 0 else Q
        return self.position - closest_point_to_landing_zone
        
    def render(self, screen):
        # Calculate the position of the landing legs
        left_landing_leg = self.position - Vector2D(0, self.LANDING_LEG_LENGTH).rotate(self.angle+self.LANDING_LEG_ANGLE)
        right_landing_leg = self.position - Vector2D(0, self.LANDING_LEG_LENGTH).rotate(self.angle-self.LANDING_LEG_ANGLE)

        pygame.draw.line(screen,
                        (255, 0, 0) if self.left_leg_collision else (255, 255, 255),
                        world_to_screen(self.position),
                        world_to_screen(left_landing_leg), 2)
        pygame.draw.line(screen,
                        (255, 0, 0) if self.right_leg_collision else (255, 255, 255),
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
                       world_to_screen(self.distance_sensors_collisions[i]),
                       5)

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
                         (Vector2D(0, WORLD_HEIGHT), Vector2D(WORLD_WIDTH, WORLD_HEIGHT)),
                         (Vector2D(WORLD_WIDTH, 0), Vector2D(WORLD_WIDTH, WORLD_HEIGHT))])
        
    def collisions_check(self, lander):
        lander.right_leg_collision = False
        lander.left_leg_collision = False
        lander.body_collision = False
        lander.distance_sensors_values = [99999 for _ in range(lander.num_distance_sensors)]
        lander.distance_sensors_collisions = [None for _ in range(lander.num_distance_sensors)]
        for i, segment in enumerate(self.segments):
            self.collisions[i] = self.collides_with_lander(lander, segment)
            for j, distance_sensor_angle in enumerate(lander.distance_sensors_angles):
                ds_segment = (lander.position, lander.position + Vector2D(WORLD_WIDTH, 0).rotate(distance_sensor_angle))
                hit, collision_point = self.collides_with_segment(segment, ds_segment)
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

        rleg_segment = (lander.position, lander.position - Vector2D(0, lander.LANDING_LEG_LENGTH).rotate(lander.angle - lander.LANDING_LEG_ANGLE))
        lleg_segment = (lander.position, lander.position - Vector2D(0, lander.LANDING_LEG_LENGTH).rotate(lander.angle + lander.LANDING_LEG_ANGLE))

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
            if min(p1.y,p2.y) <= y <= max(p1.y,p2.y) and min(p4.y,p3.y) <= y <= max(p4.y,p3.y):
                if min(p3.x,p4.x) <= p1.x <= max(p3.x,p4.x):
                    return True, Vector2D(p1.x, y)
        elif abs(p3.x - p4.x) < .0001:  # segment 2 is vertical
            # Find the intersection of segment 2 with segment 1
            m1 = (p2.y - p1.y) / (p2.x - p1.x)  # slope of segment 1
            b1 = p1.y - m1 * p1.x  # y-intercept of segment 1
            y = m1 * p3.x + b1  # y-coordinate of intersection
            # Check if the intersection point is on both segments
            if min(p1.y,p2.y) <= y <= max(p1.y,p2.y) and min(p4.y,p3.y) <= y <= max(p4.y,p3.y):
                if min(p1.x,p2.x) <= p3.x <= max(p1.x,p2.x):
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
            #print(lander)
            if collided:
                line_color = (255, 0, 0)
            elif i == self.landing_zone_index:
                line_color = (0, 255, 0)
            pygame.draw.line(screen, line_color,
                             world_to_screen(start),
                             world_to_screen(end), 2)
            i+=1
        
class Gene:
    def __init__(self, thrust, angle):
        self.thrust = thrust
        self.angle = angle

class Chromosome:
    def __init__(self, lander, genes):
        self.genes = genes
        self.fitness = 0
        self.run(lander)
        self.lander = lander

    @classmethod
    def random_chromosome(cls, lander, num_timesteps):
        genes = []
        prev_angle = random.randint(-15, 15)
        rand_angles = np.random.randint(-15, 15, size=num_timesteps)
        rand_thrust = np.random.choice([2, 3, 4], p=[.1, .1, .8], size=num_timesteps)
        for t, a in zip(rand_thrust, rand_angles):
            thrust = int(t)
            angle = int(a) + prev_angle
            angle = max(-90, angle)
            angle = min(90, angle)
            genes.append(Gene(thrust, angle))
            prev_angle = angle
        chrom = cls(lander, genes)
        return chrom

    def mutate(self, probability=.1):
        prev_gene = None
        changed = False
        for i, gene in enumerate(self.genes):
            if random.uniform(0, 1) < probability:
                changed = True
                rand_thrust = int(np.random.choice([2, 3, 4], p=[.1, .1, .8]))
                gene.thrust = rand_thrust
                rand_angle = int(np.random.randint(-15, 15))
                if i == 0:
                    gene.angle = rand_angle
                else:
                    gene.angle = rand_angle + prev_gene.angle
            prev_gene = gene
        
        if changed:
            self.run(self.lander)
    
    def run(self, lander):
        # Reset the lander to its initial state
        lander.reset()
        # Set the path of the lander to the genes in this chromosome
        self.path = lander.run(self)
        self.landed = lander.landed
        self.crashed = lander.crashed

        self.distance_from_landing_zone = lander.distance_to_landing_zone()
        self.crash_speed = copy.deepcopy(lander.velocity)
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        # Calculate the fitness based on the distance from the crash zone
        # and the length of the run

        # normalize the distance to ~0-1
        distance_from_landing_zone = (self.distance_from_landing_zone / WORLD_WIDTH / 2 )**2
        crash_speed_x = abs(self.crash_speed.x) 
        crash_speed_y = abs(self.crash_speed.y)

        fitness = 700

        if self.distance_from_landing_zone < 10:
            fitness = 850
            fitness -= crash_speed_x  + crash_speed_y 

        fitness -= distance_from_landing_zone * 500
        
        # If the lander crashed or went out of bounds, give it a low fitness
        if self.landed:
            fitness += 1000
        if self.crashed:
            fitness -= 500
        return fitness


    def render(self, screen):
        for p1, p2 in zip(self.path, self.path[1:]):
            pygame.draw.line(screen, (0, 0, 255),
                             world_to_screen(Vector2D(p1[0], p1[1])),
                             world_to_screen(Vector2D(p2[0], p2[1])), 2)

class NeuralNetworkChromosome:
    def __init__(self, lander, neural_network):
        self.nn = neural_network
        self.lander = lander

    @classmethod
    def random_chromosome(cls, lander, nn_topology):
        input_size, hidden_size1, hidden_size2, output_size = nn_topology
        neural_network = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
        return NeuralNetworkChromosome(lander, neural_network)

    def calc_fitness(self):
        # Calculate the fitness based on the distance from the crash zone
        # and the length of the run

        # normalize the distance to ~0-1
        distance_from_landing_zone = (self.distance_from_landing_zone / WORLD_WIDTH / 2 )**2
        crash_speed_x = abs(self.crash_speed.x) 
        crash_speed_y = abs(self.crash_speed.y)

        fitness = 700

        if self.distance_from_landing_zone < 10:
            fitness = 850
            fitness -= crash_speed_x  + crash_speed_y 

        fitness -= distance_from_landing_zone * 500
        
        # If the lander crashed or went out of bounds, give it a low fitness
        if self.landed:
            fitness += 1000
        if self.crashed:
            fitness -= 500
        return fitness

    def run(self):
        lander = self.lander
        # Reset the lander to its initial state
        lander.reset()
        # Set the path of the lander to the genes in this chromosome
        self.path = lander.run_neural_network(self)
        self.landed = lander.landed
        self.crashed = lander.crashed

        self.distance_from_landing_zone = lander.distance_to_landing_zone()
        self.crash_speed = copy.deepcopy(lander.velocity)
        self.fitness = self.calc_fitness()

    def mutate(self, mutation_probability=.05):
        mutation_mask_1 = np.random.binomial(1, mutation_probability, size=self.nn.weights_input_hidden1.shape)
        mutation_mask_2 = np.random.binomial(1, mutation_probability, size=self.nn.weights_hidden1_hidden2.shape)
        mutation_mask_3 = np.random.binomial(1, mutation_probability, size=self.nn.weights_hidden2_output.shape)
        
        mutation_mask_b1 = np.random.binomial(1, mutation_probability, size=self.nn.biases_hidden1.shape)
        mutation_mask_b2 = np.random.binomial(1, mutation_probability, size=self.nn.biases_hidden2.shape)
        mutation_mask_b3 = np.random.binomial(1, mutation_probability, size=self.nn.biases_output.shape)
        
        self.weights_input_hidden1 = np.where(mutation_mask_1,
                                              np.random.normal(0, np.sqrt(2 / self.input_size), self.nn.weights_input_hidden1.shape),
                                              self.weights_input_hidden1)
        self.weights_hidden1_hidden2 = np.where(mutation_mask_2,
                                                np.random.normal(0, np.sqrt(2 / self.hidden_size1), self.nn.weights_hidden1_hidden2.shape),
                                                self.weights_hidden1_hidden2)
        self.weights_hidden2_output = np.where(mutation_mask_3,
                                               np.random.normal(0, np.sqrt(2 / self.hidden_size2), self.nn.weights_hidden2_output.shape),
                                               self.weights_hidden2_output)
        
        self.biases_hidden1 = np.where(mutation_mask_b1,
                                       np.random.normal(0, np.sqrt(2 / self.input_size), self.nn.biases_hidden1.shape),
                                       self.biases_hidden1)
        self.biases_hidden2 = np.where(mutation_mask_b2,
                                       np.random.normal(0, np.sqrt(2 / self.hidden_size1), self.nn.biases_hidden2.shape),
                                       self.biases_hidden2)
        self.biases_output = np.where(mutation_mask_b3,
                                      np.random.normal(0, np.sqrt(2 / self.hidden_size2), self.nn.biases_output.shape),
                                      self.biases_output)
        self.run()

    def forward(self, inp):
        output = self.nn.forward(inp) 
        e_x = np.exp(output - np.max(output))
        #print("OUTPUT", output, "MAX", np.max(output))
        output_thrust, output_angle = e_x / e_x.sum(axis=0)
        
        #print(e_x, output_thrust, output_angle)
        # map the output to valid thrust and angle values:
        output_thrust = int(np.round(np.exp(output_thrust) / (np.exp(output_thrust) + np.exp(0))))
        output_angle = np.clip(int(round(output_angle * 180)), -90, 90)
        
        return output_thrust, output_angle
    
    def render(self, screen):
        for p1, p2 in zip(self.path, self.path[1:]):
            pygame.draw.line(screen, (0, 0, 255),
                             world_to_screen(Vector2D(p1[0], p1[1])),
                             world_to_screen(Vector2D(p2[0], p2[1])), 2)

class NeuralNetworkPopulation:
    def __init__(self, lander, size):
        self.NN_TOPOLOGY = (2+2+2+1+1+lander.num_distance_sensors,
                            10, 10, 2)
        self.chromosomes = [NeuralNetworkChromosome.random_chromosome(lander, self.NN_TOPOLOGY) for _ in range(size)]
        self.best_chromosome = self.chromosomes[0]
        self.generation_num = 0
        
class Population:
    def __init__(self, lander, size):
        self.NUM_TIMESTEPS = 200
        self.chromosomes = [Chromosome.random_chromosome(lander, self.NUM_TIMESTEPS) for _ in range(size)]
        self.best_chromosome = self.chromosomes[0]
        self.generation_num = 0

    def crossover(self, parents):
        children = []
        desired_length = len(self.chromosomes) - len(parents)
        while len(children) < desired_length:
            male = random.randint(0, len(parents) - 1)
            female = random.randint(0, len(parents) - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = random.randint(1, max(len(male.path), len(female.path)))
                child_genes = male.genes[:half] + female.genes[half:]
                child = Chromosome(lander, child_genes)
                children.append(child)
        return children

    def select(self, retain_probability=.2, random_select_probability=.7):
        retain_length = int(len(self.chromosomes) * retain_probability)
        parents = self.chromosomes[:retain_length]
        for chromosome in self.chromosomes[retain_length:]:
            if random_select_probability > random.uniform(0, 1):
                parents.append(chromosome)
        return parents

    def select1(self, retain_probability=.3):
        fits = [f.fitness for f in self.chromosomes]
        normalized_fits = [(f-min(fits))/(max(fits)-min(fits)) for f in fits]
        
        parents = random.choices(self.chromosomes,
                             weights=normalized_fits,
                             k=int(len(self.chromosomes) * retain_probability))
        return parents

    def evolve(self, lander, retain_probability=.5, mutation_probability=.01, num_elites=3, num_immigrants=10):
        self.generation_num += 1
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        if self.chromosomes[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = copy.deepcopy(self.chromosomes[0])

        # Selection
        #parents = self.select(retain_probability, random_select_probability)
        parents = self.select(retain_probability)

        # Elitism (Keep the top performing chromosomes)
        parents.extend(self.chromosomes[:num_elites])
        
        # Immigration (Introduce completely new chromosomes)
        parents.extend([Chromosome.random_chromosome(lander, self.NUM_TIMESTEPS) for _ in range(num_immigrants)])
        
        # Crossover
        children = self.crossover(parents)

        # Mutation
        for child in children:
            child.mutate(mutation_probability)
        
        parents.extend(children)
        self.chromosomes = parents

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Initialize the weights and biases for the hidden layers and output layer
        self.weights_input_hidden1 = np.random.normal(0, np.sqrt(2 / self.input_size), (self.input_size, self.hidden_size1))
        self.biases_hidden1 = np.random.normal(0, np.sqrt(2 / self.input_size), self.hidden_size1)
        self.weights_hidden1_hidden2 = np.random.normal(0, np.sqrt(2 / self.hidden_size1), (self.hidden_size1, self.hidden_size2))
        self.biases_hidden2 = np.random.normal(0, np.sqrt(2 / self.hidden_size1), self.hidden_size2)
        self.weights_hidden2_output = np.random.normal(0, np.sqrt(2 / self.hidden_size2), (self.hidden_size2, self.output_size))
        self.biases_output = np.random.normal(0, np.sqrt(2 / self.hidden_size2), self.output_size)        
    def forward(self, inp):
        # Propagate the input through the first hidden layer using the weights and biases
        hidden1 = np.dot(inp, self.weights_input_hidden1) + self.biases_hidden1
        # Apply the ReLU activation function to the first hidden layer output
        hidden1 = np.maximum(hidden1, 0)
        
        # Propagate the output of the first hidden layer through the second hidden layer using the weights and biases
        hidden2 = np.dot(hidden1, self.weights_hidden1_hidden2) + self.biases_hidden2
        # Apply the ReLU activation function to the second hidden layer output
        hidden2 = np.maximum(hidden2, 0)

        # Propagate the output of the second hidden layer through the output layer using the weights and biases
        output = np.dot(hidden2, self.weights_hidden2_output) + self.biases_output

        #output = 1 / (1 + np.exp(-output))

        return output

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

class RendererHandler:
    def __init__(self, lander, surface, poplation, sim_paused = True, keyboard_control_mode=True):
        self.population = population
        self.lander = lander
        self.surface = surface

        self.sim_paused = sim_paused
        self.sim_running = True
        self.show_debug = True
        
        pygame.font.init()
        self.debug_font = pygame.font.SysFont('Monospace', 12)
        self.pause_text = self.debug_font.render(f"SIM PAUSED", False, (235, 0, 0))
        self.debug_info_text = self.debug_font.render(f"PRESS F3 TO SHOW DEBUG INFO", False, (255, 255, 255))
        self.desired_angle = 0
        self.desired_thrust = 0
        self.keyboard_control_mode = keyboard_control_mode
        self.show_debug = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.sim_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.desired_angle = 15
                elif event.key == pygame.K_RIGHT:
                    self.desired_angle = -15
                elif event.key == pygame.K_UP:
                    self.desired_thrust = 4
                elif event.key == pygame.K_DOWN:
                    self.desired_thrust = 0
                elif event.key == pygame.K_SPACE:
                    self.sim_paused = not self.sim_paused
                elif event.key == pygame.K_F5:
                    # Reset the lander and surface
                    lander.reset()
                    surface.reset()
                elif event.key == pygame.K_F3:
                    self.show_debug = not self.show_debug
            elif event.type == pygame.KEYUP:
                # Reset the desired angle and thrust when the keys are released
                self.desired_angle = 0
                self.desired_thrust = 0
            
    def render_debug(self):
        if self.show_debug:
            if self.keyboard_control_mode:
                pos_debug_text = self.debug_font.render(f"POSITION {int(lander.position.x):>6}", False, (255, 255, 255))
                alt_debug_text = self.debug_font.render(f"ALTITUDE {int(lander.position.y):>6}", False, (255, 255, 255))

                velx_debug_text = self.debug_font.render(f"HORIZONTAL SPEED {int(lander.velocity.x):>6}", False, (255, 255, 255))
                velyt_debug_text = self.debug_font.render(f"VERTICAL   SPEED {int(lander.velocity.y):>6}", False, (255, 255, 255))

                landed_text = self.debug_font.render(f"LANDED {lander.landed}", False, (255, 255, 255))
                crashed_text = self.debug_font.render(f"CRASHED {lander.crashed}", False, (255, 255, 255))

                screen.blit(pos_debug_text, (10, 10))
                screen.blit(alt_debug_text, (10, 20))
                screen.blit(velx_debug_text, (300, 10))
                screen.blit(velyt_debug_text, (300, 20))
                screen.blit(landed_text, (1250, 20))
                screen.blit(crashed_text, (1250, 30))
            else:
                generation_text = self.debug_font.render(f"GENERATION {population.generation_num:>6}", False, (255, 255, 255))
                generation_ptext = self.debug_font.render(f"POPULATION {len(self.population.chromosomes):>6}", False, (255, 255, 255))
                generation_btext = self.debug_font.render(f"BEST FITNESS {self.population.best_chromosome.fitness:<6}", False, (255, 255, 255))
                fits = [c.fitness for c in self.population.chromosomes]
                generation_avgtext = self.debug_font.render(f"AVG FITNESS {sum(fits)//len(fits):<6}", False, (255, 255, 255))

                screen.blit(generation_text, (1250, 40))
                screen.blit(generation_ptext, (1250, 50))
                screen.blit(generation_btext, (1250, 60))
                screen.blit(generation_avgtext, (1250, 70))
        else:
            screen.blit(self.debug_info_text, (1250, 20))
            
        if self.sim_paused:
            screen.blit(self.pause_text, (1250, 10))

# Set up the Pygame window
WORLD_WIDTH, WORLD_HEIGHT = 7000, 3000
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 700
#START_X, START_Y = 1000, 2600
START_X, START_Y = 6500, 2600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('2D Mars Lander')

# Create a lander
lander = Lander(START_X, START_Y, 0, 0)
#surface_topology = [(0, 100), (1000, 500), (1500, 1500), (3000, 1000), (4000, 150), (5500, 150), (6999, 800)])
surface_topology = [(0, 1800), (300, 1200), (1000, 1550), (2000, 1200), (2500, 1650), (3700, 220), (4700, 220), (4750, 1000), (4700, 1650), (4000, 1700), (3700, 1600), (3750, 1900), (4000, 2100), (4900, 2050), (5100, 1000), (5500, 500), (6200, 800), (6999, 600)]
#surface_topology = [(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600), (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)]
surface = Surface(surface_topology)
lander.add_to_surface(surface)

clock = pygame.time.Clock()
c = [(0, -44), (4, -42), (4, -57), (4, -58), (4, -60), (4, -69), (0, -61), (4, -76), (4, -85), (4, -90), (0, -85), (4, -72), (4, -58), (4, -61), (4, -55), (3, -70), (4, -68), (4, -64), (3, -79), (0, -82), (4, -70), (4, -56), (0, -66), (3, -74), (3, -72), (4, -71), (0, -81), (4, -68), (0, -55), (0, -49), (3, -34), (4, -49), (4, -53), (4, -65), (3, -62), (4, -77), (4, -87), (4, -85), (0, -90), (0, -86), (4, -90), (4, -79), (0, -82), (0, -77), (3, -81), (0, -66), (0, -64), (4, -64), (4, -54), (4, -54), (4, -41), (4, -48), (3, -45), (4, -54), (4, -55), (0, -51), (0, -37), (0, -51), (0, -56), (4, -42), (4, -30), (4, -23), (4, -23), (0, -23), (4, -17), (0, -28), (3, -15), (4, -28), (4, -18), (4, -18), (4, -31), (4, -36), (3, -47), (4, -59), (4, -63), (3, -54), (4, -45), (4, -32), (4, -33), (4, -37), (4, -31), (4, -41), (0, -38), (4, -40), (4, -29), (4, -37), (4, -26), (4, -37), (3, -38), (3, -48), (0, -60), (0, -59), (4, -73), (4, -81), (4, -90), (4, -90), (4, -90), (0, -88), (0, -86), (4, -75), (4, -76), (4, -90), (3, -90), (4, -98), (0, -79), (4, -69), (4, -55), (4, -43), (4, -36), (4, -48), (4, -52), (0, -66), (0, -77), (4, -72), (4, -65), (4, -78), (4, -90), (3, -85), (4, -73), (4, -68), (0, -56), (3, -48), (3, -41), (4, -44), (4, -47), (4, -59), (4, -70), (3, -38), (3, -44), (0, -54), (4, -50), (0, -65), (4, -51), (3, -61), (0, -71), (4, -82), (4, -86), (4, -90), (0, -90), (4, -87), (3, -83), (3, -90), (0, -78), (0, -72), (4, -61), (4, -71), (4, -79), (4, -64), (4, -74), (4, -61), (4, -47), (4, -55), (4, -66), (4, -63), (4, -53), (3, -61), (3, -70), (3, -62), (4, -75), (0, -76), (3, -84), (4, -90), (0, -90), (3, -77), (4, -77), (4, -67), (0, -77), (4, -87), (0, -79), (0, -77), (4, -74), (4, -79), (4, -71), (4, -73), (4, -84), (4, -79), (0, -74), (4, -66), (4, -69), (0, -61), (0, -54), (3, -58), (4, -65), (4, -54), (4, -47), (4, -57), (0, -70), (3, -66), (4, -72), (4, -66), (4, -81), (3, -90), (4, -79), (4, -73), (0, -75), (3, -75), (3, -74), (4, -81), (3, -90), (4, -90)]
#[(0, -44), (4, -42), (4, -57), (4, -58), (4, -60), (4, -69), (0, -61), (4, -76), (4, -85), (4, -90), (0, -85), (4, -72), (4, -58), (4, -61), (4, -55), (3, -70), (4, -68), (4, -64), (3, -79), (0, -82), (4, -70), (4, -56), (0, -66), (3, -74), (3, -72), (4, -71), (0, -81), (4, -68), (0, -55), (0, -49), (3, -34), (4, -49), (4, -53), (4, -65), (3, -62), (4, -77), (4, -87), (4, -85), (0, -90), (0, -86), (4, -90), (4, -79), (0, -82), (0, -77), (3, -81), (0, -66), (0, -64), (4, -64), (4, -54), (4, -54), (4, -41), (4, -48), (3, -45), (4, -54), (4, -55), (0, -51), (0, -37), (0, -51), (0, -56), (4, -42), (4, -30), (4, -23), (4, -23), (0, -23), (4, -17), (0, -28), (3, -15), (4, -28), (4, -18), (4, -18), (4, -31), (4, -36), (3, -47), (4, -59), (4, -63), (3, -54), (4, -45), (4, -32), (4, -33), (4, -37), (4, -31), (4, -41), (0, -38), (4, -40), (4, -29), (4, -37), (4, -26), (4, -37), (3, -38), (3, -48), (0, -60), (0, -59), (4, -73), (4, -81), (4, -90), (4, -90), (4, -90), (0, -88), (0, -86), (4, -75), (4, -76), (4, -90), (3, -90), (4, -98), (0, -79), (4, -69), (4, -55), (4, -43), (4, -36), (4, -48), (4, -52), (0, -66), (0, -77), (4, -72), (4, -65), (4, -78), (4, -90), (3, -85), (4, -73), (4, -68), (0, -56), (3, -48), (3, -41), (4, -44), (4, -47), (4, -59), (4, -70), (3, -38), (3, -44), (0, -54), (4, -50), (0, -65), (4, -51), (3, -61), (0, -71), (4, -82), (4, -86), (4, -90), (0, -90), (4, -87), (3, -83), (3, -90), (0, -78), (0, -72), (4, -61), (4, -71), (4, -79), (4, -64), (4, -74), (4, -61), (4, -47), (4, -55), (4, -21), (4, -29), (4, -24), (4, -21), (4, -16), (4, -17), (0, -10), (4, -16), (3, -8), (4, -23), (3, -8), (4, -12), (0, -22), (4, -23), (4, -38), (4, -43), (4, -52), (4, -59), (3, -70), (0, -61), (4, -49), (3, -53), (4, -62), (4, -52), (4, -47), (0, -55), (4, -53), (3, -55), (4, -56), (0, -50), (4, -49), (4, -45), (0, -41), (4, -34), (3, -21), (4, -28), (4, -36), (4, -29), (4, -19), (3, -31), (4, -21), (4, -12), (0, -13), (4, -12), (3, -9), (4, -2), (4, 7), (3, 17)]
i = 0
population = Population(lander, 100)
#population = None
# Run the game loop
renderer_handler = RendererHandler(lander, surface, population, sim_paused=False, keyboard_control_mode=False)
running = True
nnch = NeuralNetworkChromosome.random_chromosome(lander, (2+2+2+1+1+lander.num_distance_sensors,20, 20, 2))
while running:
    # step into either the population sim or the lander sim
    if not renderer_handler.sim_paused:
        if renderer_handler.keyboard_control_mode:
            DT = 1 # time step in seconds
            #print(nnch.forward(lander.get_state()))
            t, a= nnch.forward(lander.get_state())
#            renderer_handler.lander.step(a, t, DT)
            renderer_handler.lander.step(renderer_handler.desired_angle, renderer_handler.desired_thrust, DT)
#            renderer_handler.lander.step(c[i][0], c[i][1], .5)
#            renderer_handler.lander.step(c[i][0], c[i][1], .5)
#            renderer_handler.lander.step(c[i][0], c[i][1], DT)
            i+=1
        else:
            population.evolve(lander)
    # Clear the screen
    screen.fill((0, 0, 0))
    # Handle events
    renderer_handler.handle_events()
    renderer_handler.render_debug()
    
    if not renderer_handler.sim_paused:
        if renderer_handler.keyboard_control_mode:
            # Render the lander and surface
            lander.render(screen)
            surface.render(screen)
            time.sleep(.1)
        else:
            lander.reset()
            surface.reset()
            lander.render(screen)
            surface.render(screen)
            # Render the surface
            
            for chrom in population.chromosomes:
                if chrom.path[-1][1] < 0:
                    print([(gene.thrust, gene.angle)for gene in chrom.genes])
                chrom.render(screen)
            print(len(population.best_chromosome.path), len(population.chromosomes))
    else:
        lander.render(screen)
        surface.render(screen)
        
    # Update the display
    pygame.display.update()

