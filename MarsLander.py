import math
import pygame
import time
import random
import numpy as np
import copy

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

    def apply_force(self, force):
        self.acceleration += force

    def add_to_surface(self, surface):
        self.surface = surface

    def step(self, desired_angle, desired_thrust, dt):
        # Check if the lander has landed or gone out of bounds
        if self.position.y > WORLD_HEIGHT or self.position.y < 0 or self.crashed or self.landed or self.position.x < 0 or self.position.x > WORLD_WIDTH:
            # Stop the lander
            return False

        # Update the force and angle of the thruster
        angle_change = desired_angle - self.angle
        self.angle += -15 if angle_change <= -15 else 15 if angle_change >= 15 else angle_change
        self.force = desired_thrust
        # Calculate the force vector of the thruster
        thruster_force = Vector2D(0, self.force).rotate(self.angle)
        # Apply the forces of the thruster and gravity
        self.apply_force(thruster_force)
        self.apply_force(Vector2D(0, -self.GRAVITY))
        # Update the velocity of the lander based on its acceleration
        self.velocity += self.acceleration
        # Update the position of the lander based on its velocity
        self.position += self.velocity * dt
        # Reset the acceleration of the lander
        self.acceleration *= 0
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

class Surface:
    def __init__(self, points):
        self.points = points
        self.collisions = []
        self.segments = []
        # Generate the segments from the points
        for i in range(len(points) - 1):
            segment = (Vector2D(points[i]), Vector2D(points[i + 1]))
            self.segments.append(segment)
            if segment[0].y == segment[1].y:
                self.landing_zone_index = i
            self.collisions.append(False)

    def collisions_check(self, lander):
        lander.right_leg_collision = False
        lander.left_leg_collision = False
        lander.body_collision = False
        for i, segment in enumerate(self.segments):
            self.collisions[i] = self.collides_with_lander(lander, segment)
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

        rleg_segment = (lander.position, lander.position - Vector2D(0, lander.LANDING_LEG_LENGTH).rotate(lander.angle-lander.LANDING_LEG_ANGLE))
        lleg_segment = (lander.position, lander.position - Vector2D(0, lander.LANDING_LEG_LENGTH).rotate(lander.angle+lander.LANDING_LEG_ANGLE))

        rleg_collision = self.collides_with_segment(segment, rleg_segment)
        lleg_collision = self.collides_with_segment(segment, lleg_segment)
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
        """if projection >= 0 and projection <= length:
            proj_point = p1 + projection/length * (p2 - p1)
            pygame.draw.line(screen, (255, 0, 0),
                             world_to_screen(proj_point),
                             world_to_screen(point), 2)
            pygame.draw.circle(screen,
                           (0, 255, 0),
                           world_to_screen(proj_point),
                           5)"""
        # Check if the projection falls outside the segment
        if projection < 0:
            return (point - p1).length()
        elif projection > length:
            return (point - p2).length()

        # Calculate the distance between the point and the segment
        return (point - (p1 + projection / length * (p2 - p1))).length()
    
    def collides_with_segment(self, segment_1, segment_2):
        # Convert the line segments to the form y = mx + b
        p1, p2 = segment_1
        p3, p4 = segment_2
        m1 = (p2.y - p1.y) / (p2.x - p1.x) if p2.x != p1.x else float('inf')
        b1 = p1.y - m1 * p1.x
        m2 = (p4.y - p3.y) / (p4.x - p3.x) if p4.x != p3.x else float('inf')
        b2 = p3.y - m2 * p3.x
        #x = (b2 - b1) / (m1 - m2)
        #y = m1 * x + b1
        #pygame.draw.circle(screen,
        #       (0, 255, 0),
        #       world_to_screen(Vector2D(x,y)),
        #       5)
        # Check if the segments are colliding
        if m1 != m2:
            x = (b2 - b1) / (m1 - m2)
            #print((x >= p1.x and x <= p2.x) or (x >= p2.x and x <= p1.x),
            #      (x >= p3.x and x <= p4.x) or (x >= p4.x and x <= p3.x),
            #       (b2 - b1) / (m1 - m2), segment_1, segment_2)
            # Calculate the x-coordinate of the intersection point
            # Check if the x-coordinate falls within both segments
            if (x >= p1.x and x <= p2.x) or (x >= p2.x and x <= p1.x):
                if (x >= p3.x and x <= p4.x) or (x >= p4.x and x <= p3.x):
                    return True
        else:
            # Check if the y-intercepts are equal (coincident)
            if b1 == b2:
                return True
            # Check if the segments overlap
            elif (p3.x >= p1.x and p3.x <= p2.x) or (p3.x >= p2.x and p3.x <= p1.x):
                return True
        return False

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
        for t, a in zip(np.random.choice([0, 1, 2, 3, 4], size=num_timesteps-1, p=[.1, .1, .1, .1, .6]),
                        np.random.randint(-90, 90, size=num_timesteps-1)):
            thrust = int(t)
            angle = int(a)
            genes.append(Gene(thrust, angle))
        genes.append(Gene(4, 0))
        chrom = cls(lander, genes)
        chrom.run(lander)
        return chrom

    def mutate(self, probability=.1):
        for i, gene in enumerate(self.genes):
            if random.uniform(0, 1) < probability and i != len(self.genes) -1:
                gene.thrust = random.choices([0, 1, 2, 3, 4], cum_weights=[1, 1, 1, 1, 6], k=1)[0]
                gene.angle = random.randint(-90, 90)
            
    def run(self, lander):
        # Reset the lander to its initial state
        lander.reset()
        # Set the path of the lander to the genes in this chromosome
        self.path = lander.run(self)
        self.landed = lander.landed
        self.crashed = lander.crashed
        landing_zone_segment = lander.surface.segments[lander.surface.landing_zone_index]
        landing_zone_midpoint = (landing_zone_segment[0] + landing_zone_segment[1])/2
        self.distance_from_crash_zone = (lander.position - landing_zone_midpoint).length()
        self.crash_speed = copy.deepcopy(lander.velocity)
        self.fitness = self.calc_fitness()
    def calc_fitness(self):
        # Calculate the fitness based on the distance from the crash zone
        # and the length of the run
        distance_from_crash_zone = self.distance_from_crash_zone
        fitness = 500 - distance_from_crash_zone / 20 \
                      - (0 if abs(self.crash_speed.x) < 20 else abs(20 - abs(self.crash_speed.x))) \
                      - (0 if abs(self.crash_speed.y) < 40 else abs(40 - abs(self.crash_speed.y))) \
        #- len(self.path)
        # If the lander crashed or went out of bounds, give it a low fitness
        if self.landed:
            fitness = 1000
        return fitness

    def render(self, screen):
        for p1, p2 in zip(self.path, self.path[1:]):
            pygame.draw.line(screen, (0, 0, 255),
                             world_to_screen(Vector2D(p1[0], p1[1])),
                             world_to_screen(Vector2D(p2[0], p2[1])), 2)

class Population:
    def __init__(self, lander, size):
        NUM_TIMESTEPS = 200
        self.chromosomes = [Chromosome.random_chromosome(lander, NUM_TIMESTEPS) for _ in range(size)]
        self.best_chromosome = self.chromosomes[0]
    def evolve(self, lander, retain_probability=.5, random_select_probability=.7, mutation_probability=.5):
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
                half = len(male.genes) // 2
                child_genes = male.genes[:half] + female.genes[half:]
                child = Chromosome(lander, child_genes)
                child.mutate(mutation_probability)
                children.append(child)
        parents.extend(children)
        self.chromosomes = parents

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
# Set up the Pygame window
WORLD_WIDTH, WORLD_HEIGHT = 7000, 3000
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('2D Mars Lander')
pygame.font.init()
debug_font = pygame.font.SysFont('Monospace', 12)
# Create a lander
lander = Lander(6500, 2600, 0, 0)
surface = Surface([(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600), (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)])
lander.add_to_surface(surface)

clock = pygame.time.Clock()
simulation_speed_factor = 1
pause_sim = True
pause_text = debug_font.render(f"SIM PAUSED", False, (235, 0, 0))
desired_angle = 0
desired_thrust = 0
"""
population = Population(lander, 100)
#population.evolve(lander)
winning_path = None
generation_num = 0
# Run the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause_sim = not pause_sim
            elif event.key == pygame.K_F5:
                lander.reset()
                surface = Surface([(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600), (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)])
                lander.add_to_surface(surface)
        elif event.type == pygame.KEYUP:
            desired_angle = 0
            desired_thrust = 0
    # Clear the screen
    screen.fill((0, 0, 0))

    pos_debug_text = debug_font.render(f"POSITION {int(lander.position.x):>6}", False, (255, 255, 255))
    alt_debug_text = debug_font.render(f"ALTITUDE {int(lander.position.y):>6}", False, (255, 255, 255))

    velx_debug_text = debug_font.render(f"HORIZONTAL SPEED {int(lander.velocity.x):>6}", False, (255, 255, 255))
    velyt_debug_text = debug_font.render(f"VERTICAL   SPEED {int(lander.velocity.y):>6}", False, (255, 255, 255))

    landed_text = debug_font.render(f"LANDED {lander.landed}", False, (255, 255, 255))
    crashed_text = debug_font.render(f"CRASHED {lander.crashed}", False, (255, 255, 255))
    generation_text = debug_font.render(f"GENERATION {generation_num:<6}", False, (255, 255, 255))
    generation_ptext = debug_font.render(f"POPULATION {len(population.chromosomes):<6}", False, (255, 255, 255))
    generation_btext = debug_font.render(f"BEST FITNESS {population.best_chromosome.fitness:<6}", False, (255, 255, 255))

    screen.blit(pos_debug_text, (10, 10))
    screen.blit(alt_debug_text, (10, 20))
    screen.blit(velx_debug_text, (300, 10))
    screen.blit(velyt_debug_text, (300, 20))
    screen.blit(landed_text, (1250, 20))
    screen.blit(crashed_text, (1250, 30))
    screen.blit(generation_text, (1250, 40))
    screen.blit(generation_ptext, (1250, 50))
    screen.blit(generation_btext, (1250, 60))
    if pause_sim:
        screen.blit(pause_text, (1300, 10))
    population.evolve(lander)

    # Render the surface
    surface.render(screen)
    for chrom in population.chromosomes:
        chrom.render(screen)
        if 0:
            winning_chromosome = chrom
            running = False
        lander.reset()
    # Render the lander
    lander.render(screen)

    # Update the display
    pygame.display.update()

    generation_num += 1
lander.reset()

for gene in population.best_chromosome.genes:
    lander.step(gene.angle, gene.thrust, 1)

    # Clear the screen
    screen.fill((0, 0, 0))

    pos_debug_text = debug_font.render(f"POSITION {int(lander.position.x):>6}", False, (255, 255, 255))
    alt_debug_text = debug_font.render(f"ALTITUDE {int(lander.position.y):>6}", False, (255, 255, 255))

    velx_debug_text = debug_font.render(f"HORIZONTAL SPEED {int(lander.velocity.x):>6}", False, (255, 255, 255))
    velyt_debug_text = debug_font.render(f"VERTICAL   SPEED {int(lander.velocity.y):>6}", False, (255, 255, 255))

    landed_text = debug_font.render(f"LANDED {lander.landed}", False, (255, 255, 255))
    crashed_text = debug_font.render(f"CRASHED {lander.crashed}", False, (255, 255, 255))
    generation_text = debug_font.render(f"GENERATION {generation_num:<6}", False, (255, 255, 255))
    generation_ptext = debug_font.render(f"POPULATION {len(population.chromosomes):<6}", False, (255, 255, 255))

    screen.blit(pos_debug_text, (10, 10))
    screen.blit(alt_debug_text, (10, 20))
    screen.blit(velx_debug_text, (300, 10))
    screen.blit(velyt_debug_text, (300, 20))
    screen.blit(landed_text, (1300, 20))
    screen.blit(crashed_text, (1300, 30))
    screen.blit(generation_text, (1300, 40))
    screen.blit(generation_ptext, (1300, 50))
    
    # Render the surface
    surface.render(screen)

    #Render the lander
    lander.render(screen)
    time.sleep(.5)
    pygame.display.update()
    
"""

running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Update the angle and force of the thruster based on key presses
            if event.key == pygame.K_LEFT:
                desired_angle = 60
            elif event.key == pygame.K_RIGHT:
                desired_angle = -60
            elif event.key == pygame.K_UP:
                desired_thrust = 8
            elif event.key == pygame.K_DOWN:
                desired_thrust = 0
            elif event.key == pygame.K_SPACE:
                pause_sim = not pause_sim
            elif event.key == pygame.K_F5:
                lander.reset()
                surface = Surface([(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600), (2200, 1550), (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750)])
                lander.add_to_surface(surface)

                clock = pygame.time.Clock()
        elif event.type == pygame.KEYUP:
            desired_angle = 0
            desired_thrust = 0

    # Get the elapsed time since the last frame
    elapsed_time = clock.tick(60) / 1000
    if not pause_sim:  
        # Step the simulation
        lander.step(desired_angle, desired_thrust, 1)#elapsed_time * simulation_speed_factor)
    time.sleep(.1)
    # Clear the screen
    screen.fill((0, 0, 0))

    pos_debug_text = debug_font.render(f"POSITION {int(lander.position.x):>6}", False, (255, 255, 255))
    alt_debug_text = debug_font.render(f"ALTITUDE {int(lander.position.y):>6}", False, (255, 255, 255))

    velx_debug_text = debug_font.render(f"HORIZONTAL SPEED {int(lander.velocity.x):>6}", False, (255, 255, 255))
    velyt_debug_text = debug_font.render(f"VERTICAL   SPEED {int(lander.velocity.y):>6}", False, (255, 255, 255))

    landed_text = debug_font.render(f"LANDED {lander.landed}", False, (255, 255, 255))
    crashed_text = debug_font.render(f"CRASHED {lander.crashed}", False, (255, 255, 255))
    
    screen.blit(pos_debug_text, (10, 10))
    screen.blit(alt_debug_text, (10, 20))
    screen.blit(velx_debug_text, (300, 10))
    screen.blit(velyt_debug_text, (300, 20))
    screen.blit(landed_text, (1300, 20))
    screen.blit(crashed_text, (1300, 30))
    if pause_sim:
        screen.blit(pause_text, (1300, 10))
    # Render the surface
    surface.render(screen)

    # Render the lander
    lander.render(screen)

    # Update the display
    pygame.display.update()

