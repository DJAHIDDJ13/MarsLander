from multiprocessing import Pool
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import math
import random
from typing import List, Tuple


# https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return np.array(points, dtype='int32')


def draw_polygon(ax, polygon):
    # Create a rectangle with width 7000 and height 3000
    ax.set_xlim([0, 7000])
    ax.set_ylim([0, 3000])

    polygon = np.vstack((polygon, polygon[0]))
    # Plot the polygon
    x, y = polygon.T
    ax.plot(x, y, 'k-', linewidth=2)


# Using Angle summation method, terrible speed, because of the arccos and no vectorization
# Still wrong sometimes, not gonna bother debugging it since i know it's slow
def is_point_inside_polygon_AS(point, polygon):
    angle_sum = 0
    for i in range(len(polygon)):
        p1, p2 = polygon[i], polygon[(i + 1) % len(polygon)]
        ba = p1 - point
        bc = p2 - point
        dot_product = np.dot(ba, bc)
        ba_magnitude = np.linalg.norm(ba)
        bc_magnitude = np.linalg.norm(bc)

        ccw = 1 if np.cross(ba, bc) > -1e-9 else -1

        angle_sum += ccw * \
            np.arccos(dot_product / (ba_magnitude * bc_magnitude))
    return abs(angle_sum - 2*np.pi) < 1e-9

# Raycasting method, ray is casted to the right of the point
# Already x7 faster than the angle summation method
# Vectorized, gained an additional x3 for a total of ~x20 faster than AS
# Still has problems when they y of the point is equal to one of
# the ends of the polygon vertices' y, but isn't really an issue if point is a float


def is_point_inside_polygon_RC(point, polygon_segments):
    points_aa, points_ab = polygon_segments
    points_ba = np.tile(point, (len(points_aa[0]), 1))
    # setting the segment to just right of the map
    points_bb = np.tile(np.array([7001, point[1]]),
                        (len(points_aa[0]), 1))

    x1, y1 = points_aa
    x2, y2 = points_ab
    dx1, dy1 = x2 - x1, y2 - y1

    x3, y3 = points_ba.T
    x4, y4 = points_bb.T
    dx2, dy2 = x4 - x3, y4 - y3

    # Calculate the determiants
    det = dx1 * dy2 - dy1 * dx2

    # Calculate the parameter values for the intersection point
    # div by zero retuns -inf and raises warning, no problems in calculation however
    # Might be worth it to supress the warning temporarily here,
    # using a np.where instead seems overkill
    x3_x1 = x3-x1
    y1_y3 = y1-y3
    t1 = (dy2 * x3_x1 + dx2 * y1_y3) / det
    t2 = (dy1 * x3_x1 + dx1 * y1_y3) / det

    collisions = (t1 >= 0) & (t1 <= 1) & (t2 >= 0) & (t2 <= 1)

    return collisions.sum() % 2 == 1


""" 
# Parallelization at the vertex level is not worth it.
# terrible overhead because i have very few vertices
def check_point_in_polygon(args):
    return is_point_inside_polygon_RC(*args)

def is_point_inside_polygon_RC_(point, polygon_segments, num_cores=8):
    chunks = [(point, (a.T, b.T))
              for a, b in zip(np.array_split(polygon_segments[0].T, num_cores),
                              np.array_split(polygon_segments[1].T, num_cores))]
    with Pool(processes=num_cores) as pool:
        results = pool.map(check_point_in_polygon, chunks)

    return sum(results) % 2 == 1
"""


class PointEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, polygon_segments):
        self.polygon_segments = polygon_segments

    def transform(self, X):
        # X is a 2D array with shape (n_samples, 2)
        # apply point_side_of_line_vec to each point and segment
        # to get the side for each segment for each point
        sides = [PointEncoder.point_side_of_line_vec(
            x, self.polygon_segments) for x in X]
        # stack the results along the third axis and return
        return sides

    def fit(self, X, y=None):
        # no fitting necessary
        return self

    @staticmethod
    def point_side_of_line_vec(point, polygon_segments):
        points_aa, points_ab = polygon_segments

        x1, y1 = points_aa
        x2, y2 = points_ab

        dx = x2 - x1
        dy = y2 - y1

        x_diff = point[0] - x1
        y_diff = point[1] - y1

        cross = dx * y_diff - dy * x_diff

        return cross > 0


def main():
    NUM_POLYS = 1

    NUM_POINTS = 100000
    polys = []
    for _ in range(NUM_POLYS):
        polys.append(generate_polygon(center=(3000, 1500),
                                      avg_radius=1000,
                                      irregularity=0.35,
                                      spikiness=0.4,
                                      num_vertices=40))

    #polys = [np.array([[0, 0], [100, 100], [0, 100]])]
    # polys = [np.array([(0, 450), (300, 750), (1000, 450), (1500, 650), (1800, 850), (2000, 1950), (2200, 1850), (2400, 2000), (3100, 1800), (3150, 1550), (2500, 1600), (2200, 1550),
    #                   (2100, 750), (2200, 150), (3200, 150), (3500, 450), (4000, 950), (4500, 1450), (5000, 1550), (5500, 1500), (6000, 950), (6999, 1750), (7000, 3000), (0, 3000)])]
    # Create an empty dictionary to store the benchmark times
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    draw_polygon(polygon=polys[0], ax=ax)

    times = {'RC': [], 'AS': [], 'DT': []}

    # Loop over each polygon
    for polygon in polys:
        # only needs to be calculated once at the start
        # Overhead for RC and decision tree
        start_time = time.time()
        poly_segments = (polygon.T, np.roll(polygon, -1, axis=0).T)
        rc_overhead = time.time() - start_time

        # Generate some random points
        points = np.random.uniform(
            low=[0, 0], high=[7000, 3000], size=(NUM_POINTS, 2))

        # this should be part of the overhead for the decision tree
        train_points = np.random.uniform(
            low=[0, 0], high=[7000, 3000], size=(1000, 2))

        # Benchmark the ray casting method
        colors = ['red' if is_point_inside_polygon_RC(
            point, poly_segments) else 'green' for point in points]

        train_colors = ['red' if is_point_inside_polygon_RC(
            point, poly_segments) else 'green' for point in train_points]

        pipeline = Pipeline([
            ('encoder', PointEncoder(poly_segments)),
            ('clf', DecisionTreeClassifier())
        ])
        start_time = time.time()
        # surprisingly cheap overhead but the boundaries,
        # still not worth it because the prediction time longer than RC
        # when used point by point(the way it's used in the lander)
        pipeline.fit(train_points, train_colors)
        dt_overhead = time.time() - start_time

        predictions = []
        start_time = time.time()
        for point in points:
            # it's used like this so we gotta use a loop
            pipeline.predict(point[np.newaxis, :])
        dt_time = time.time() - start_time

        ax.scatter(points[:, 0], points[:, 1], c=colors)

        start_time = time.time()
        for point in points:
            is_point_inside_polygon_RC(point, poly_segments)
        rc_time = time.time() - start_time

        """
        # Benchmark the angle summation method
        start_time = time.time()
        for point in points:
            is_point_inside_polygon_AS(point, polygon)
        as_time = time.time() - start_time
        """
        # Store the benchmark times
        times['RC'].extend([rc_overhead, rc_time])
        times['AS'].extend([0, 1])
        times['DT'].extend([dt_overhead, dt_time])

    # Print out the results
    for method in times:
        print(f'{method} method:')
        for i, polygon_time in enumerate(times[method]):
            print(f'Polygon {i+1}: {polygon_time:.6f} seconds')
        print(f'Average time: {np.mean(times[method]):.6f} seconds')
        print()
    plt.show()


if __name__ == '__main__':
    main()
