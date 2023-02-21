# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).
It gathers all visualizations of the investigated and explained planning algorithms.
License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPBenchmark import Benchmark
from IPEnvironment import CollisionChecker
from shapely.geometry import Point, Polygon, LineString
import math
import numpy as np


benchList = list()

# -----------------------------------------
trapField = dict()
trapField["obs1"] = LineString([(6, 18), (6, 8), (16, 8), (16, 18)]).buffer(1.0)
description = "Following the direct connection from goal to start would lead the algorithm into a trap."
benchList.append(Benchmark("Trap", CollisionChecker(trapField), [[10, 15]], [[10, 1]], description, 2))

# -----------------------------------------
bottleNeckField = dict()
bottleNeckField["obs1"] = LineString([(0, 13), (11, 13)]).buffer(.5)
bottleNeckField["obs2"] = LineString([(13, 13), (23, 13)]).buffer(.5)
description = "Planer has to find a narrow passage."
benchList.append(Benchmark("Bottleneck", CollisionChecker(bottleNeckField), [[4, 15]], [[18, 1]], description, 2))

# -----------------------------------------
fatBottleNeckField = dict()
fatBottleNeckField["obs1"] = Polygon([(0, 8), (11, 8), (11, 15), (0, 15)]).buffer(.5)
fatBottleNeckField["obs2"] = Polygon([(13, 8), (24, 8), (24, 15), (13, 15)]).buffer(.5)
description = "Planer has to find a narrow passage with a significant extend."
benchList.append(Benchmark("Fat bottleneck", CollisionChecker(
    fatBottleNeckField), [[4, 21]], [[18, 1]], description, 2))

# -----------------------------------------
trapField = dict()
for i in range(10, 1300, 10):
    radius = 1.0 * (i / 500.0)
    width = 1.0 * (i / 5000.0)
    trapField["obsA"+str(i/10)] = Point([(10 - np.cos(np.deg2rad(i))*radius,
                                          10 - np.sin(np.deg2rad(i))*radius)]).buffer(width)
    trapField["obsB"+str(i/10)] = Point([(15 + np.sin(np.deg2rad(i))*radius,
                                          15 + np.cos(np.deg2rad(i))*radius)]).buffer(width)
trapField["obsC"] = LineString([(5, 0.5), (5, 10), (15, 20), (20, 20)]).buffer(0.5)

start = [[10, 10]]
goal = [[15, 15]]

description = "Two spirals block the way from start to goal."
benchList.append(Benchmark("Spirals", CollisionChecker(trapField), start, goal, description, 4))


# -----------------------------------------
trapField = dict()
form1 = Point(11.5, 11.5).buffer(10.0)
form2 = LineString([(0, 11.5), (11.5, 11.5), (16, 5), (16, 18)]).buffer(1.0)

trapField["obs1"] = form1.difference(form2)

start = [[1, 5]]
goal = [[16, 18]]

description = "The robot needs to find the entrance to the circle."
benchList.append(Benchmark("Entrance", CollisionChecker(trapField), start, goal, description, 2))

# -----------------------------------------
trapField = dict()

for i in range(1, 9):
    circle = Point(11.5, 11.5).buffer(1.0 * i)
    innercircle = Point(11.5, 11.5).buffer((1.0 * i) - 0.5)
    entrance = LineString([(11.5, 11.5), (11.5 + np.sin(np.deg2rad(360.0 / i) * (-1)**i) * (1.0 * i),
                          11.5 + np.cos(np.deg2rad(360.0 / i) * (-1)**i) * (1.0 * i))]).buffer(0.2)
    ring = circle.difference(innercircle)
    trapField["obs"+str(i)] = ring.difference(entrance)

start = [[5, 5]]
goal = [[11.5, 11.5]]

description = "The robot needs to find all entrances to the circles."
benchList.append(Benchmark("Entrances", CollisionChecker(trapField), start, goal, description, 2))

# -----------------------------------------
scene = dict()
scene['K'] = Polygon([(0, 8), (0, 17), (3, 17), (3, 13.5), (5.5, 17), (9, 17),
                     (5.5, 12.5), (9, 8), (5.5, 8), (3, 11.5), (3, 8)])
scene['I'] = Polygon([(9.5, 8), (9.5, 17), (12.5, 17), (12.5, 8)])
scene['T'] = Polygon([(16, 8), (16, 14), (13, 14), (13, 17), (22, 17), (22, 14), (19, 14), (19, 8)])

description = 'Not an really serious challenge.'
benchList.append(Benchmark('KIT', CollisionChecker(scene), [[1, 1]], [[21, 21]], description, 1))

# -----------------------------------------
scene = dict()
line = LineString([(0, 0), (0, 22), (22, 22), (22, 0), (0, 0)]).buffer(2.0)
scene['rim'] = line.buffer(-1.0)
line = LineString([(6, 6), (6, 16), (16, 16), (16, 6), (6, 6)]).buffer(0.75)
scene['obs'] = line.buffer(-0.25)

description = 'A rounded rectangle inside a rounded rectangle..'
benchList.append(Benchmark('Inside', CollisionChecker(scene), [[3, 3]], [[19, 19]], description, 1))

# -----------------------------------------
scene = dict()
scene['obs0'] = LineString([(0, 11.5), (6.33, 11.5)]).buffer(1.0)
scene['obs1'] = LineString([(22, 11.5), (15.66, 11.5)]).buffer(1.0)
scene['obs2'] = LineString([(11, 5), (11, 17)]).buffer(1.0)
scene['obs3'] = LineString([(5.5, 0), (5.5, 6)]).buffer(1.0)
scene['obs4'] = LineString([(16.5, 0), (16.5, 6)]).buffer(1.0)
scene['obs5'] = LineString([(5.5, 22), (5.5, 17)]).buffer(1.0)
scene['obs6'] = LineString([(16.5, 22), (16.5, 17)]).buffer(1.0)

description = 'A simple smetic labyrinth.'
benchList.append(Benchmark('SSL', CollisionChecker(scene), [[1, 1]], [[21, 21]], description, 2))

# -----------------------------------------
scene = dict()
a = Point(11, 11).buffer(8)
b = Point(11, 11).buffer(7.5)
scene['obs'] = a.difference(b)

description = 'A simple ring.'
benchList.append(Benchmark('Ring', CollisionChecker(scene), [[4, 4]], [[18, 18]], description, 2))

# -----------------------------------------
scene = dict()
a = Point(5.25, 5.25).buffer(5.5)
b = Point(4.75, 5.25).buffer(5.5)
scene['obs0'] = a.difference(b)
a = Point(5.25, 16.75).buffer(5.5)
b = Point(4.75, 16.75).buffer(5.5)
scene['obs1'] = a.difference(b)
a = Point(16.75, 5.25).buffer(5.5)
b = Point(17.25, 5.25).buffer(5.5)
scene['obs2'] = a.difference(b)
a = Point(16.75, 16.75).buffer(5.5)
b = Point(17.25, 16.75).buffer(5.5)
scene['obs3'] = a.difference(b)
a = Point(9, 11).buffer(5.5)
b = Point(9.5, 11).buffer(5.5)
scene['obs4'] = a.difference(b)
a = Point(13, 11).buffer(5.5)
b = Point(12.5, 11).buffer(5.5)
scene['obs5'] = a.difference(b)

description = 'Six Hemispheres have to be mastered.'
benchList.append(Benchmark('Hemispheres', CollisionChecker(scene), [[1, 1]], [[21, 21]], description, 3))

# -----------------------------------------
simpleField = dict()
simpleField["obs1"] = LineString([(12.5, 0), (12.5, 15)]).buffer(1.0)
simpleField["obs2"] = LineString([(7, 15), (18, 15)]).buffer(1.0)
simpleDescription = "Around the hammerhead"
benchList.append(Benchmark("Hammerhead", CollisionChecker(simpleField), [[10, 1]], [[15, 1]], simpleDescription, 1))


# -----------------------------------------
mediumField = dict()
mediumField["obs1"] = LineString([(0, 20), (15, 20)]).buffer(1.0)
mediumField["obs2"] = LineString([(0, 10), (15, 10)]).buffer(1.0)
mediumField["obs3"] = LineString([(10, 15), (25, 15)]).buffer(1.0)
mediumField["obs4"] = LineString([(10, 5), (25, 5)]).buffer(1.0)
mediumDescription = "Through the zigzag"
benchList.append(Benchmark("Zigzag", CollisionChecker(mediumField), [[12.5, 24]], [[12.5, 1]], mediumDescription, 2))


# -----------------------------------------
# Compute spiral points to add to line string
def spiralPoints(center=(12.5, 12.5), radius=10, numPoints=30, coils=4):
    points = []
    awayStep = float(radius)/float(numPoints)
    aroundStep = float(coils)/float(numPoints)
    aroundRadians = aroundStep * 2 * math.pi
    rotation = math.pi
    for i in range(numPoints):
        away = i * awayStep
        around = i * aroundRadians + rotation
        x = center[0] + math.cos(around) * away
        y = center[1] + math.sin(around) * away
        points.append((x, y))
    return points


hardField = dict()
hardField["obs1"] = LineString(spiralPoints(center=(12.5, 12.5), radius=10, numPoints=300, coils=4)).buffer(0.1)
hardDescription = "Through the spiral"
benchList.append(Benchmark("Spiral", CollisionChecker(hardField), [[12.5, 24]], [[13.1, 12.9]], hardDescription, 3))


# -----------------------------------------
scene = dict()
scene["obs1"] = LineString([(20, 25), (20, 20), (24, 20)]).buffer(0.1)
scene["obs2"] = LineString([(25, 19), (19, 19), (19, 24)]).buffer(0.1)
scene["obs3"] = LineString([(1, 5), (5, 5), (5, 0)]).buffer(0.1)
scene["obs4"] = LineString([(6, 1), (6, 6), (0, 6)]).buffer(0.1)
scene["obs5"] = LineString([(19, 24), (5, 7)]).buffer(0.1)
scene["obs6"] = LineString([(1, 15), (20, 15)]).buffer(0.1)
description = 'Medium challenge. Large free space with narrow passings'
benchList.append(Benchmark('medium', CollisionChecker(scene), [[1, 1]], [[24, 24]], description, 1))

# -----------------------------------------
mediumField = dict()
pol = Polygon([(17, 0), (17, 25), (25, 25), (25, 0)]).buffer(1.0)
line = LineString([(17.5, 22.5), (22.5, 17.5), (17.5, 12.5), (22.5, 12.5), (17.5, 7.5), (22.5, 2.5)]).buffer(0.75)
mediumField["obs1"] = pol.difference(line)

description = 'Medium challenge. Large free space with only one narrow way '
benchList.append(Benchmark('medium2', CollisionChecker(mediumField), [[17.5, 22.5]], [[22.5, 2.5]], description, 1))

# -----------------------------------------
scene = dict()

scene["top"] = Polygon([[5, 25], [5, 10], [10, 10], [10, 20], [15, 20], [15, 25]])
scene["bottom"] = Polygon([[10, 0], [10, 5], [15, 5], [15, 15], [20, 15], [20, 0]])

description = 'Easy challenge'
benchList.append(Benchmark('L-square', CollisionChecker(scene), [[5, 5]], [[20, 20]], description, 1))

# -----------------------------------------
scene = dict()

scene["teeth_bottom"] = Polygon([(7.5, 0), (10, 15), (12.5, 0), (15, 15), (17.5, 0)])
scene["tooth_top"] = Polygon([(10, 25), (12.5, 10), (15, 25)])

description = 'Medium challenge'
benchList.append(Benchmark('Teeth', CollisionChecker(scene), [[2, 2]], [[23, 2]], description, 2))

# -----------------------------------------
scene = dict()

gearshift_polygon = []
gearshift_x = 2
gearshift_y = 3
slot_count = 10
slot_width = 1
slot_length = 8
slot_spacing = 1
gearshift_width = slot_spacing + slot_count * (slot_width + slot_spacing)
gearshift_heigth = 2 * slot_spacing + 2 * slot_length + slot_width

gearshift_polygon.append((gearshift_x, gearshift_y))
gearshift_polygon.append((gearshift_x + gearshift_width, gearshift_y))
gearshift_polygon.append((gearshift_x + gearshift_width, gearshift_y + gearshift_heigth))
gearshift_polygon.append((gearshift_x, gearshift_y + gearshift_heigth))
gearshift_polygon.append((gearshift_x, gearshift_y + slot_spacing + slot_length + slot_width))

# upper slots
for i in range(0, slot_count):
    slot_start_x = gearshift_x + slot_spacing + i * (slot_width + slot_spacing)
    slot_start_y = gearshift_y + slot_spacing + slot_length + slot_width
    gearshift_polygon.append((slot_start_x, slot_start_y))
    gearshift_polygon.append((slot_start_x, slot_start_y + slot_length))
    gearshift_polygon.append((slot_start_x + slot_width, slot_start_y + slot_length))
    gearshift_polygon.append((slot_start_x + slot_width, slot_start_y))

# lower slots
for i in range(slot_count - 1, -1, -1):
    slot_start_x = gearshift_x + slot_spacing + i * (slot_width + slot_spacing) + slot_width
    slot_start_y = gearshift_y + slot_spacing + slot_length
    gearshift_polygon.append((slot_start_x, slot_start_y))
    gearshift_polygon.append((slot_start_x, slot_start_y - slot_length))
    gearshift_polygon.append((slot_start_x - slot_width, slot_start_y - slot_length))
    gearshift_polygon.append((slot_start_x - slot_width, slot_start_y))

gearshift_polygon.append((gearshift_x, gearshift_y + slot_spacing + slot_length))

scene["gearshift"] = Polygon(gearshift_polygon)

description = 'Hard challenge'
benchList.append(Benchmark('Gearshift', CollisionChecker(scene), [[24, 2]], [[21.5, 4.5]], description, 3))

# -----------------------------------------


def rectangle(x, y, width, height=None):
    if height is None:
        height = width
    return Polygon([(x, y+height), (x+width, y+height), (x + width, y), (x, y)])


def zigzag(x, y, length):
    baseX = x
    baseY = y
    x_coords = [baseX + i for i in range(length)]
    y_coords = [baseY + (i % 2) for i in range(length)]
    return LineString(zip(x_coords, y_coords)).buffer(.2)


benchmark_easy = dict()
benchmark_easy["obs1"] = rectangle(5, 10, 3)
benchmark_easy["obs2"] = rectangle(10, 15, 3)
benchmark_easy["obs3"] = rectangle(10, 5, 3)
benchmark_easy["obs4"] = rectangle(15, 10, 3)
description = "Four squares to avoid"
#limits = [[0,0],[23,23]]
start = [11.5, 22]
goal = [11.5, 2]
difficulty = 1
benchList.append(Benchmark("Squares_easy", CollisionChecker(benchmark_easy), [start], [goal], description, difficulty))

benchmark_medium = dict()
benchmark_medium["obs1"] = zigzag(5, 18, 15)
benchmark_medium["obs2"] = zigzag(0, 12, 10)
benchmark_medium["obs3"] = zigzag(15, 12, 10)
benchmark_medium["obs4"] = zigzag(5, 5, 15)
description = "Four zigzag lines to avoid"
#limits = [[0,0],[23,23]]
start = [11.5, 21]
goal = [11.5, 2]
difficulty = 2
benchList.append(Benchmark("Zigzag_medium", CollisionChecker(
    benchmark_medium), [start], [goal], description, difficulty))

benchmark_hard = dict()
for i in range(0, 22, 8):
    benchmark_hard["row_a" + str(i)] = zigzag(0, i + 4, 18)
    benchmark_hard["row_b" + str(i)] = zigzag(8, i, 18)
#limits = [[0,0],[23,23]]
description = "Many zigzag lines to avoid"
start = [22, 22]
goal = [1, 0.8]
difficulty = 3
benchList.append(Benchmark("Zigzag_hard", CollisionChecker(benchmark_hard), [start], [goal], description, difficulty))

# -----------------------------------------
japanField = dict()
japanField["obs1"] = Point([12, 12]).buffer(5.0)
description = "Easy field with circle"
benchList.append(Benchmark("Japan", CollisionChecker(japanField), [[18, 13]], [[6, 10]], description, 2))


# -----------------------------------------
haystackField = dict()
middleCircle = [12, 12]
for i in range(9):
    haystackField["obs1"+str(i)] = LineString([(21-2*i, 21), middleCircle]).buffer(0.05)
    haystackField["obs2"+str(i)] = LineString([(1+2*i, 2), middleCircle]).buffer(0.05)
    haystackField["obs3"+str(i)] = LineString([(2, 21-2*i), middleCircle]).buffer(0.05)
    haystackField["obs4"+str(i)] = LineString([(21, 1+2*i), middleCircle]).buffer(0.05)
description = "Needle in a haystack"
benchList.append(Benchmark("Haystack", CollisionChecker(haystackField), [[5, 10.5]], [[19, 11.7]], description, 3))

# -----------------------------------------
hairPerson = dict()
middleCircle = [12, 12]
hairPerson["obs1"] = Point(middleCircle).buffer(5.0)
hairPerson["obs2"] = Polygon([(7, 0), (7, 6), (17, 6), (17, 0)]).buffer(1.0)
for i in range(10):
    hairPerson["obs3"+str(i)] = LineString([(21-2*i, 21), middleCircle]).buffer(0.2)
description = "From Shoulder to shoulder"
benchList.append(Benchmark("HairPerson", CollisionChecker(hairPerson), [[9, 7.5]], [[15, 7.5]], description, 4))
