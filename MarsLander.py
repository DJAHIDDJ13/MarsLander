import math
import pygame
import time
import random
import numpy as np
import copy
import gym

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
            self.__init__(6500, 2600, 0, 0)

    def run(self, chromosome):
        path = []
        for gene in chromosome.genes:
            self.step(gene.angle, gene.thrust, 1)
            path.append([lander.position.x, lander.position.y, lander.velocity.x, lander.velocity.y])
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
        
        crash_pos = lander.position
        self.crash_pos = crash_pos
        landing_zone_segment = lander.surface.segments[lander.surface.landing_zone_index]
        line_of_sight_segment = (crash_pos, (landing_zone_segment[0] + landing_zone_segment[1]) / 2)
        self.line_of_sight = True
        crashes = []
        for i, segment in enumerate(lander.surface.segments):
            if i != lander.surface.landing_zone_index:
                if lander.surface.collides_with_segment(line_of_sight_segment, segment)[0]:
                    crashes += [i]
                    self.line_of_sight = False
                    break
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
            fitness = 1000
            fitness -= crash_speed_x  + crash_speed_y 

        fitness -= distance_from_landing_zone * 500
        
        # Check line of sight from crash zone to landing zone
        if self.line_of_sight:
            # If there's a line of sight, give a bonus to the fitness
            fitness += 500

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

class Population:
    def __init__(self, lander, size):
        NUM_TIMESTEPS = 150
        self.chromosomes = [Chromosome.random_chromosome(lander, NUM_TIMESTEPS) for _ in range(size)]
        self.best_chromosome = self.chromosomes[0]
        self.generation_num = 0

    def selection2(self, retain_probability=.5, random_select_probability=.7):
        # Calculate the total fitness of all the chromosomes
        total_fitness = sum(chromosome.fitness for chromosome in self.chromosomes)
        
        # Normalize the fitness values to be between 0 and 1
        normalized_fitnesses = [(chromosome.fitness - min(self.chromosomes, key=lambda x: x.fitness).fitness) / (1000 - (-2000))
                                for chromosome in self.chromosomes]
        
        # Calculate the probability of selecting each chromosome
        selection_probabilities = [fitness / total_fitness for fitness in normalized_fitnesses]
        
        # Sort the chromosomes by fitness
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        
        if self.chromosomes[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = copy.deepcopy(self.chromosomes[0])
            
        # Select the chromosomes to be retained
        parents = []
        for chromosome, probability in zip(self.chromosomes, selection_probabilities):
            if random_select_probability > random.uniform(0, 1):
                parents.append(chromosome)
            retain_probability -= probability
            if retain_probability < 0:
                break
        
        return parents
    
    def selection(self, retain_probability=.5, random_select_probability=.7):
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        if self.chromosomes[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = copy.deepcopy(self.chromosomes[0])
        retain_length = int(len(self.chromosomes) * retain_probability)
        parents = self.chromosomes[:retain_length]
        for chromosome in self.chromosomes[retain_length:]:
            if random_select_probability > random.uniform(0, 1):
                parents.append(chromosome)
        return parents

    def crossover(self, parents, children_size):
        children = []
        desired_length = children_size - len(parents)
        while len(children) < desired_length:
            male = random.randint(0, len(parents) - 1)
            female = random.randint(0, len(parents) - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = len(male.genes) // 2
                child_genes = male.genes[:half] + female.genes[half:]
                children.append(Chromosome(lander, child_genes))
        return children

    def evolve2(self, lander, retain_probability=.4, random_select_probability=.5, mutation_probability=.3):
        self.generation_num += 1
        parents = self.selection(retain_probability, random_select_probability)
        children = self.crossover(parents, len(self.chromosomes))
        for child in children:
            child.mutate(mutation_probability)
        parents.extend(children)
        self.chromosomes = parents

    def evolve(self, lander, retain_probability=.25, random_select_probability=.3, mutation_probability=.4):
        self.generation_num += 1
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        if self.chromosomes[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = copy.deepcopy(self.chromosomes[0])
        retain_length = int(len(self.chromosomes) * retain_probability)
        parents = self.chromosomes[:retain_length]
        for chromosome in self.chromosomes[retain_length:]:
            if random_select_probability > random.uniform(0, 1):
                parents.append(chromosome)
        children = []
        desired_length = len(self.chromosomes) - len(parents)
        while len(children) < desired_length:
            male = random.randint(0, len(parents) - 1)
            female = random.randint(0, len(parents) - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = (len(male.path) + len(female.path)) // 4
                if self.generation_num > 50:
                    female.mutate(.6)
                child_genes = male.genes[:half] + female.genes[half:]
                child = Chromosome(lander, child_genes)
                child.mutate(mutation_probability)
                children.append(child)
        parents.extend(children)
        self.chromosomes = parents

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the weights and biases for the hidden and output layers
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.biases_hidden = np.random.randn(self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.biases_output = np.random.randn(self.output_size)

    def forward(self, input):
        # Propagate the input through the hidden layer using the weights and biases
        hidden = np.dot(input, self.weights_input_hidden) + self.biases_hidden
        # Apply the ReLU activation function to the hidden layer output
        hidden = np.maximum(hidden, 0)

        # Propagate the hidden layer output through the output layer using the weights and biases
        output = np.dot(hidden, self.weights_hidden_output) + self.biases_output
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
    def __init__(self, lander, surface, poplation, keyboard_control_mode=True):
        self.population = population
        self.lander = lander
        self.surface = surface

        self.sim_paused = True
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
                    self.desired_angle = 60
                elif event.key == pygame.K_RIGHT:
                    self.desired_angle = -60
                elif event.key == pygame.K_UP:
                    self.desired_thrust = 8
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

class MarsLanderEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    def __init__(self, init_position, init_velocity, surface_topology, render_mode='human'):
        self.lander = Lander(init_position, init_velocity)
        self.surface = Surface(surface_topology)
        self.lander.add_to_surface(self.surface)
        low_pos = np.array([0, 0], dtype=np.float32)
        high_pos = np.array([7000, 3000], dtype=np.float32)
        low_vel = np.array([-1000, -1000], dtype=np.float32)
        high_vel = np.array([1000, 1000], dtype=np.float32)
        low_angle = np.array([-90], dtype=np.float32)
        high_angle = np.array([90], dtype=np.float32)
        low_thrust = np.array([0], dtype=np.float32)
        high_thrust = np.array([4], dtype=np.float32)
        low_sensors = np.zeros(self.lander.num_distance_sensors, dtype=np.float32)
        high_sensors = np.full(self.lander.num_distance_sensors, 7000, dtype=np.float32)
        low = np.concatenate((low_pos, low_vel, low_angle, low_thrust, low_sensors))
        high = np.concatenate((high_pos, high_vel, high_angle, high_thrust, high_sensors))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0, -90]), high=np.array([4, 90]), dtype=np.int32)

        self.screen = None
        self.render_mode = render_mode
        self.clock = None
        
    def reset(self):
        self.lander.reset()
        self.surface.reset()
        self.lander.add_to_surface(self.surface)

        if self.render_mode == "human":
            self.render()
        return self.lander.get_state()
    
    def step(self, action):
        self.lander.step(action[1], action[0])
        pos = self.lander.position
        vel = self.lander.velocity
        state = [
            pos.x,
            pos.y,
            vel.x,
            vel.y,
            self.lander.angle,
            self.lander.force
        ]
        state.extend(self.lander.distance_sensors_values)

        assert len(state) == 6 + self.lander.num_distance_sensors
        
        reward = 3000
        landing_zone_segment = self.surface.segments[self.surface.landing_zone_index]
        landing_zone_midpoint = (landing_zone_segment[0] + landing_zone_segment[1])/2
        distance_from_landing_zone = (self.lander.position - landing_zone_midpoint).length()
        reward -= distance_from_landing_zone

        done = self.lander.landed or self.lander.crashed
        info = {}

        if self.render_mode == "human":
            self.render()
        return state, reward, done, info

    def render(self, mode='human'):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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

# Set up the Pygame window
WORLD_WIDTH, WORLD_HEIGHT = 7000, 3000
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 700
# Create a lander
SURFACE_TOPOLOGY = [(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600), (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)]
INIT_POSITION = Vector2D(6500, 2600)
INIT_VELOCITY = Vector2D(0, 0)

lander_env = MarsLanderEnv(INIT_POSITION, INIT_VELOCITY, SURFACE_TOPOLOGY, render_mode='human')
lander_env.render()
desired_angle, desired_thrust = 0, 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.sim_running = False
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
                lander.reset()
                surface.reset()
        elif event.type == pygame.KEYUP:
            # Reset the desired angle and thrust when the keys are released
            desired_angle = 0
            desired_thrust = 0
    action = [desired_thrust, desired_angle]
    observation, reward, done, info = lander_env.step(action)
    #print(observation, reward, done, info)
    lander_env.render()
    if done:
        break

lander_env.close()
