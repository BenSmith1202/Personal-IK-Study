import numpy as np
import pyglet
from pyglet import shapes
from pyglet.gl import *
import random
from math import dist, sqrt
from pyglet.window import key
from time import time
import copy
from statistics import mean


class BHTree:
    def __init__(self, parent, cornerx, cornery, width):
        self.parent = parent
        self.children = [None, None, None, None]
        self.cornerx = cornerx  # location of the quadrant
        self.cornery = cornery
        self.width = width  # width of the quadrant
        self.bodies = []
        self.mass = 0
        self.cmx = 0.0
        self.cmy = 0.0


class GraphWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bh_tree_root = None
        self.main_batch = pyglet.graphics.Batch()
        self.overlay_batch = pyglet.graphics.Batch()
        self.body_list = []  # the list of CBody objects
        self.graph_line_list = []  # the list of shapes.Line objects
        self.tree_line_list = []  # the list of shapes.Line objects
        self.drawing_graph = False
        self.drawing_tree = False
        self.output_runtime = False
        self.using_BH = False
        self.is_paused = False
        self.draw_counter = 0
        self.draw_divider = 1  # leave alone

    def on_draw(self):
        self.draw_counter += 1
        if self.draw_counter % self.draw_divider == 0:  # framerate limiter (has issues, W.I.P.)
            self.clear()
            self.main_batch.draw()

    def update(self, dt: float):  # physics update. run rate controlled in main
        if not self.is_paused:
            self.graph_line_list = []  # clear line list
            self.tree_line_list = []
            self.bh_tree_root = build_sim_tree(None, self.body_list)
            NBodySim.update_bodies(self,
                                   self.body_list)  # update all bodies according to the direct-n-body simulation approach

            if self.drawing_tree:
                draw_tree(self, self.bh_tree_root)  # could be vastly sped up by not drawing every frame

            if self.drawing_graph:  # draws lines to make a fully connected graph between the bodies
                for body in self.body_list:
                    for body2 in self.body_list:
                        self.graph_line_list.append(
                            shapes.Line(body.square.x, body.square.y, body2.square.x, body2.square.y, 1,
                                        color=(64, 64, 64), batch=self.main_batch))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.N:  # Start a simulation with one stationary body
            self.main_batch = pyglet.graphics.Batch()
            self.body_list = []
            self.body_list.extend(random_body_list(self, 1, 50, 1, 1))
            # self.body_list.append(direct_sim.CBody(shapes.Rectangle(0, 0, 10, 10, batch=self.batch), 200000, (self.width/2, self.height/2), (0,0)))
            print(f"Bodies: {len(self.body_list)}")
        elif symbol == key.M:  # start a random three body simulation
            self.main_batch = pyglet.graphics.Batch()
            self.body_list = []
            self.body_list.extend(random_body_list(self, 3, 50, 2, 2))
            for body in self.body_list:
                body.velocity = (random.uniform(-1, 1), random.uniform(-1, 1))
            print(f"Bodies: {len(self.body_list)}")
        elif symbol == key._1:  # add 20 light-weight bodies
            self.body_list.extend(random_body_list(self, 500, 50, 2, 2))
            print(f"Bodies: {len(self.body_list)}")
        elif symbol == key._2:  # add 1 heavy body
            self.body_list.extend(random_body_list(self, 500, 50, 2, 2))
            print(f"Bodies: {len(self.body_list)}")
        elif symbol == key._3:  # add 1 super-heavy body
            self.body_list.extend(random_body_list(self, 1, 5000, 10, 10))
            print(f"Bodies: {len(self.body_list)}")
        # elif symbol == key.V: # add 20 bodies to a vortex
        #     self.body_list =random_vortex_list(self, 20, 50, 3, 3)
        #     print(f"Bodies: {len(self.body_list)}")
        elif symbol == key.G:  # toggle graph drawing
            self.drawing_graph = not self.drawing_graph
            print(f"Lines: {'ON' if self.drawing_graph else 'OFF'}")
        elif symbol == key.B:  # toggle tree drawing
            self.drawing_tree = not self.drawing_tree
            print(f"Tree: {'ON' if self.drawing_tree else 'OFF'}")
        elif symbol == key.H:  # toggle tree drawing
            self.using_BH = not self.using_BH
            print(f"Barnes-Hut Approximation: {'ON' if self.using_BH else 'OFF'}")
        elif symbol == key.O:  # toggle output
            self.output_runtime = not self.output_runtime
            print(f"Runtime Output: {'ON' if self.output_runtime else 'OFF'}")
        elif symbol == key.P:  # pause
            self.is_paused = not self.is_paused
            print(f"paused: {'ON' if self.is_paused else 'OFF'}")
        # testing accuracy
        elif symbol == key.T:
            print_to_txt = True

            bh_list = copy.deepcopy(self.body_list)

            self.using_BH = False
            NBodySim.update_bodies(self, self.body_list, False)
            n_positions = [body.position for body in self.body_list]

            self.using_BH = True
            NBodySim.update_bodies(self, bh_list, False)
            bh_positions = [body.position for body in bh_list]

            distances = []

            for n_pos, bh_pos in zip(n_positions, bh_positions):
                distance = [n_pos[0] - bh_pos[0], n_pos[1] - bh_pos[1]]
                magnitude = sqrt(distance[0] ** 2 + distance[1] ** 2)
                distances.append(magnitude)

            print(f"Average distance:  {mean(distances)}")

            if (print_to_txt == True):
                file_name = 'distances1.txt'

                with open(file_name, 'w') as file:
                    for distance in distances:
                        file.write(f"{distance}\n")


def random_body_list(gwindow: GraphWindow, num_bodies: int, mass: float, width: float,
                     height: float):  # creates a list of random bodies
    body_list = []
    for i in range(num_bodies):
        body_list.append(CBody(shapes.Rectangle(0, 0, width, height, batch=gwindow.main_batch),
                               random.uniform(mass, mass), [random.uniform(1, gwindow.width),
                                                            random.uniform(1, gwindow.height)],
                               [random.uniform(0, 0), random.uniform(0, 0)]))
    return body_list


def random_vortex_list(gwindow: GraphWindow, num_bodies: int, mass: float, width: float,
                       height: float):  # creates a list of random bodies
    body_list = []
    x = random.uniform(gwindow.height / 2 - gwindow.height / 4, gwindow.height / 2 + gwindow.height / 4)
    y = random.uniform(gwindow.height / 2 - gwindow.height / 4, gwindow.height / 2 + gwindow.height / 4)
    for i in range(num_bodies):
        body_list.append(CBody(shapes.Rectangle(0, 0, width, height, batch=gwindow.main_batch),
                               random.uniform(mass, mass), (x, y),
                               NBodySim.calculate_centripetal_velocity(x, y, gwindow.width / 2, gwindow.height / 2,
                                                                       0.5)))
    return body_list


def draw_tree(gwindow: GraphWindow, root_node: BHTree):
    if root_node is None:
        return
    # Draw quadrant border
    gwindow.tree_line_list.append(shapes.Line(
        root_node.cornerx, root_node.cornery,
        root_node.cornerx + root_node.width, root_node.cornery,
        width=1, color=(0, 255, 255), batch=gwindow.main_batch
    ))
    gwindow.tree_line_list.append(shapes.Line(
        root_node.cornerx + root_node.width, root_node.cornery,
        root_node.cornerx + root_node.width, root_node.cornery + root_node.width,
        width=1, color=(0, 255, 255), batch=gwindow.main_batch
    ))
    gwindow.tree_line_list.append(shapes.Line(
        root_node.cornerx + root_node.width, root_node.cornery + root_node.width,
        root_node.cornerx, root_node.cornery + root_node.width,
        width=1, color=(0, 255, 255), batch=gwindow.main_batch
    ))
    gwindow.tree_line_list.append(shapes.Line(
        root_node.cornerx, root_node.cornery + root_node.width,
        root_node.cornerx, root_node.cornery,
        width=1, color=(0, 255, 255), batch=gwindow.main_batch
    ))

    # Recursively draw children
    for quadrant in root_node.children:
        if quadrant is not None:
            draw_tree(gwindow, quadrant)


def find_bounding_square(bodies: [],
                         margin=0):  # returns the corner and side length of a square that encompasses all the bodies.
    if not bodies:
        return 0.0, 0.0, 1.0

    x_coords, y_coords = [list(coord) for coord in zip(*[body.position for body in bodies])]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Make square by taking the larger dimension, then add some space as a margin
    side_length = max(max_x - min_x, max_y - min_y) + (2 * margin)

    # Adjust corner to center the margin
    corner_x = min_x - margin
    corner_y = min_y - margin

    return corner_x, corner_y, side_length


def build_sim_tree(parent, body_list: list) -> BHTree:
    bounding_square = find_bounding_square(body_list, 20)
    root = BHTree(parent, bounding_square[0], bounding_square[1], bounding_square[2])
    build_out_branches(root, body_list)
    return root


def build_out_branches(root: BHTree, body_list: list):
    # Base case: if 1 or fewer bodies, stop recursion
    if len(body_list) <= 1:
        # If there's exactly one body, this becomes a leaf node
        if body_list:
            root.mass = body_list[0].mass
            root.cmx = body_list[0].position[0]
            root.cmy = body_list[0].position[1]
        return

    quadrants = [[] for _ in range(4)]  # 0=tl, 1=tr, 2=bl, 3=br; lists of bodies in each quadrant
    for body in body_list:
        coords = (body.position[0], body.position[1])
        if coords[0] < root.cornerx + (root.width / 2):  # if the body is on the left half of the square
            if coords[1] < root.cornery + (root.width / 2):  # if it is on the top half
                quadrants[0].append(body)
            else:  # if it is on the bottom half
                quadrants[2].append(body)
        else:  # if it is on the right half
            if coords[1] < root.cornery + (root.width / 2):  # if it is on the top half
                quadrants[1].append(body)
            else:  # if it is on the bottom half
                quadrants[3].append(body)

    total_mass = 0
    cmx = 0
    cmy = 0

    for i in range(4):  # for each quadrant
        if len(quadrants[i]) > 0:  # if quadrant has bodies
            # Quadrant creation logic remains the same
            if i == 0:
                root.children[i] = BHTree(root, root.cornerx, root.cornery, root.width / 2)  # top left quadrant
            elif i == 1:
                root.children[i] = BHTree(root, (root.cornerx + root.width / 2), root.cornery,
                                          root.width / 2)  # top right
            elif i == 2:
                root.children[i] = BHTree(root, root.cornerx, (root.cornery + root.width / 2),
                                          root.width / 2)  # bottom left
            elif i == 3:
                root.children[i] = BHTree(root, (root.cornerx + root.width / 2), (root.cornery + root.width / 2),
                                          root.width / 2)  # bottom right

            build_out_branches(root.children[i], quadrants[i])  # recurse into child quadrants

            if root.children[i] is not None:  # accumulate mass of child quadrants
                child_mass = root.children[i].mass
                total_mass += child_mass
                cmx += root.children[i].cmx * child_mass
                cmy += root.children[i].cmy * child_mass

    # Set the total mass and center of mass for this node
    if total_mass > 0:
        root.mass = total_mass
        root.cmx = cmx / total_mass
        root.cmy = cmy / total_mass

    return


# def calculate_center_of_mass(total_mass, quadrants):
#     if not quadrants or total_mass == 0:
#         return 0, 0
#     x_center = 0
#     y_center = 0
#     for quadrant in quadrants:
#         if not quadrant is None:
#             x_center += quadrant.cmx * quadrant.mass
#             y_center += quadrant.cmy * quadrant.mass
#
#     x_center /= total_mass
#     y_center /= total_mass
#
#     return x_center, y_center


# Defining Body Class
class CBody:
    def __init__(self, square: shapes.Rectangle, mass, position, velocity) -> None:
        self.square = square
        self.mass = mass
        self.position = position
        self.old_position = [position[0] - velocity[0] * NBodySim.delta_T, position[1] - velocity[1] * NBodySim.delta_T]
        self.velocity = velocity
        self.acceleration = [0, 0]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['square']  # Exclude square from deep copying
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class NBodySim:
    # defining constants
    delta_T = 0.1
    gravcon = 1
    total_mass = 0

    @staticmethod
    def bh_force(body: CBody, root: BHTree, min_distance=2.5):
        if root is None:
            return (0, 0)

        dx = root.cmx - body.position[0]
        dy = root.cmy - body.position[1]
        d = sqrt(dx ** 2 + dy ** 2)

        if d < min_distance:
            d = min_distance

        s = root.width

        theta = 0.5

        is_external = all(child is None for child in root.children)
        if (s / d) < theta or is_external:
            force_magnitude = (NBodySim.gravcon * body.mass * root.mass) / (d ** 2)
            unit_vector_x = dx / d
            unit_vector_y = dy / d
            return (force_magnitude * unit_vector_x, force_magnitude * unit_vector_y)
        else:
            total_force_x = 0
            total_force_y = 0
            for child in root.children:
                if child is not None:
                    fx, fy = NBodySim.bh_force(body, child, min_distance)
                    total_force_x += fx
                    total_force_y += fy
            return (total_force_x, total_force_y)

    # calculate the force and equal and opposite force between two bodies
    @staticmethod
    def pairwise_force(b1, b2, min_distance=2.5):
        distance = [b2.position[0] - b1.position[0], b2.position[1] - b1.position[1]]
        magnitude = sqrt(distance[0] ** 2 + distance[1] ** 2)

        # softening parameter
        effective_distance = max(magnitude, min_distance)

        # error handling
        if effective_distance == 0:
            return [0, 0], [0, 0]

        # constant representing the force of gravity
        force_magnitude = (NBodySim.gravcon * b1.mass * b2.mass) / (effective_distance ** 2)
        unit_vector = [distance[0] / effective_distance, distance[1] / effective_distance]
        # force_magnitude multiplied by the unit vector
        force = [force_magnitude * unit_vector[0], force_magnitude * unit_vector[1]]

        # return force and equal and opposite force
        return force, [-force[0], -force[1]]

    # main update function
    @staticmethod
    def update_bodies(gwindow: GraphWindow, body_list: list, square_bool=True):
        start_time = time()
        NBodySim.total_mass = 0
        # set acceleration to 0 to avoid carry-over
        for body in body_list:
            NBodySim.total_mass += body.mass
            body.acceleration = [0, 0]

        # using BH
        if gwindow.using_BH:
            for i, body_i in enumerate(body_list):
                force_i = NBodySim.bh_force(body_i, gwindow.bh_tree_root)
                body_i.acceleration[0] = force_i[0] / body_i.mass
                body_i.acceleration[1] = force_i[1] / body_i.mass
                # print(f"{force_i} force")
        # using N-Body
        else:
            delete_list = []
            new_bodies = []
            for i, body_i in enumerate(body_list):
                for j in range(i + 1, len(body_list)):
                    body_j = body_list[j]
                    force_ij, force_ji = NBodySim.pairwise_force(body_i, body_j)

                    # calculate acceleration in accordance with f=ma
                    body_i.acceleration[0] += force_ij[0] / body_i.mass
                    body_i.acceleration[1] += force_ij[1] / body_i.mass
                    body_j.acceleration[0] += force_ji[0] / body_j.mass
                    body_j.acceleration[1] += force_ji[1] / body_j.mass

                    # # Relative Mass Sizing (NOT STABLE: COMMENT OUT IF NECESSARY)
                    # for body in body_list:
                    #     body.square.width = max(2, (body.mass / NBodySim.total_mass) * 30)
                    #     body.square.height = max(2, (body.mass / NBodySim.total_mass) * 30)
                    #
                    # # Mass Merging (NOT STABLE: COMMENT OUT IF NECESSARY)
                    # if dist(body_i.position, body_j.position) < max(body_i.square.width, body_j.square.width):
                    #     new_body = NBodySim.merge_bodies(gwindow, body_i, body_j)
                    #     new_bodies.append(new_body)
                    #     delete_list.append(body_i)
                    #     delete_list.append(body_j)

            # Remove the old bodies and add the new merged bodies
            for body in delete_list:
                if body in body_list:
                    body.square = Noner
                    body_list.remove(body)
            body_list.extend(new_bodies)

        delta_T_sq = NBodySim.delta_T ** 2

        # loop to update position and velocity for each body
        for i, body in enumerate(body_list):
            # updating position and velocity using Verlet Method
            new_position = [
                2 * body.position[0] - body.old_position[0] + body.acceleration[0] * delta_T_sq,
                2 * body.position[1] - body.old_position[1] + body.acceleration[1] * delta_T_sq
            ]
            body.velocity = [
                (body.position[0] - body.old_position[0]) / (2 * NBodySim.delta_T),
                (body.position[1] - body.old_position[1]) / (2 * NBodySim.delta_T)
            ]

            # setting positions
            body.old_position = body.position
            body.position = new_position

            # set points to render squares
            if square_bool == True:
                body.square.x = body.position[0]
                body.square.y = body.position[1]

            end_time = time()
            elapsed_time = end_time - start_time
            if gwindow.output_runtime and elapsed_time > 0:
                print("Update time: ", elapsed_time, " seconds")
                print("Updates per second: " + str(1 / elapsed_time))

    @staticmethod
    def merge_bodies(gwindow: GraphWindow, body_1: CBody, body_2: CBody) -> CBody:
        width_1 = body_1.square.width
        width_2 = body_2.square.width
        combined_mass = body_1.mass + body_2.mass
        mass_ratio_1 = body_1.mass / combined_mass
        mass_ratio_2 = body_2.mass / combined_mass
        total_velocity = (
            mass_ratio_1 * body_1.velocity[0] + mass_ratio_2 * body_2.velocity[0],
            mass_ratio_1 * body_1.velocity[1] + mass_ratio_2 * body_2.velocity[1]
        )
        if body_1.mass > body_2.mass:
            position = body_1.position
        else:
            position = body_2.position

        total_width = width_1 + width_2  # unnecesary?

        return CBody(shapes.Rectangle(0, 0, total_width, total_width, batch=gwindow.main_batch), combined_mass,
                     position, total_velocity)

    @staticmethod
    def calculate_centripetal_velocity(body_x, body_y, center_x, center_y, coef):
        return -coef * (body_x - center_x), coef * (body_y - center_y)


if __name__ == '__main__':
    window = GraphWindow(width=1920, height=1080, caption="Planetary System")
    pyglet.clock.schedule_interval(window.update, 1 / 3000)
    pyglet.clock.schedule_interval(lambda dt: window.on_draw, 1 / 5)
    pyglet.app.run()

