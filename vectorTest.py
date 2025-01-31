import numpy
import numpy as np
import pyglet
from numpy.random import random
from numpy import linalg


def normalize(v):
    """Normalizes a vector to a unit vector"""
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def random_vector(length):
    """Generates a random unit vector and scales it to 'length' """
    x = random() * 2 - 1
    y = random() * 2 - 1
    return normalize(numpy.array([x, y]))*length


class VectorBone:
    """A VectorBone stores a vector, a line representing it,
    and implements some useful features like repositioning, scaling,
    and holding a set length"""

    def __init__(self, vwindow, origin, vector: np.array, length=-1):
        self.vwindow = vwindow  # the vector window to render in
        self.origin = origin    # the origin of the vector
        self.vector = vector    # the vector itself
        self.end_point = self.origin + self.vector  # the world coordinates of the tip of the vector
        if length == -1:    #if no length specified
            self.length = linalg.norm(vector)  # Calculate the length from the vector's magnitude
        else: # otherwise...
            self.length = length
            self.vector = normalize(self.vector) * length  # Normalize and scale to the specified length

        self.update_endpoint(self.origin + self.vector)    # fix the endpoint to the scaled vector

    @staticmethod
    def from_coords(vwindow, origin: tuple, end_point: tuple, length=-1):
        """Alternate constructor that makes a bone with a 'point A to point B' method"""
        origin = origin
        end_point = np.array(end_point)
        vector = end_point - origin     #same as above method, just in reverse
        return VectorBone(vwindow, origin, vector, length)

    def update_endpoint(self, end_point):
        """moves the endpoint of a bone and updates the bone accordingly"""
        self.vector = end_point - self.origin
        self.vector = normalize(self.vector) * self.length
        self.end_point = self.origin + self.vector

        # Update the existing line instead of creating a new one
        if hasattr(self, 'line'):   #make sure it actually has a line already
            self.line.x = self.origin[0]    # update line
            self.line.y = self.origin[1]
            self.line.x2 = self.end_point[0]
            self.line.y2 = self.end_point[1]
        else:
            self.line = pyglet.shapes.Line( #make a new line
                self.origin[0], self.origin[1],
                self.end_point[0], self.end_point[1],
                2, batch=self.vwindow.line_batch
            )

    def update_origin(self, new_origin):
        """Shifts the origin of a bone to a new location. and shifts the rest of the bone with it"""
        self.origin = new_origin
        self.end_point = self.origin + self.vector

        if hasattr(self, 'line'):   # see "update_endpoint"
            self.line.x = self.origin[0]
            self.line.y = self.origin[1]
            self.line.x2 = self.end_point[0]
            self.line.y2 = self.end_point[1]
        else:
            self.line = pyglet.shapes.Line(
                self.origin[0], self.origin[1],
                self.end_point[0], self.end_point[1],
                2, batch=self.vwindow.line_batch
            )


class Armature:
    """A linkage of vector bones that can use inverse kinematics to reach toward a target point"""

    def __init__(self, vwindow, vector_bones: []):
        self.bones = vector_bones   #the list of VectorBones
        self.origin = vector_bones[0].origin    # the origin of the first bone in the linkage
        self.vwindow = vwindow  # the window that will be rendering the armature

    def fabrik(self, target: (), margin=0.1):
        """Forwards and Backwards Reaching Inverse Kinematics.
        Uses an IK approximation algorithm (FABRIK) to move
        the armature towards a target point"""
        target = np.array(target)   # the target to reach for
        total_length = sum(bone.length for bone in self.bones)  # the total length of the armature
        start = np.array(self.origin)   # the origin of the first bone in the armature

        # Early exit if no movement needed
        if np.linalg.norm(self.bones[-1].end_point - target) < margin:
            return

        # if target is too far, stretch toward it
        if total_length < np.linalg.norm(target - start):   # if the target is further than the length of the armature
            direction = normalize(target - start)   # get the direction from the start to the target
            current_pos = start
            for bone in self.bones:     # lay the bones out end to end reaching as far towards the target as they can
                new_end = current_pos + direction * bone.length
                bone.update_origin(current_pos)
                bone.update_endpoint(new_end)
                current_pos = new_end
            return


        # FABRIK iteration
        iterations = 10 #iteration cap
        for _ in range(iterations):
            # BACKWARD PASS
            current_pos = target.copy()     # get a copy of the target's location
            for bone in reversed(self.bones):     # for each bone, starting from the last:
                direction = normalize(current_pos - np.array(bone.origin))  # point the bone along a line from the target
                                                                            # (or last moved bone's tip) its own origin
                new_origin = current_pos - direction * bone.length  # move the bones origin to the target
                bone.update_origin(new_origin)  # update it
                bone.update_endpoint(current_pos)
                current_pos = new_origin    # use the tip of this bone to align the next

            # FORWARD PASS
            # the armature now starts at the target and points to the origin. Running a backward pass improves the
            # accuracy and gets the armature back to its proper origin
            current_pos = start.copy()  # get a new target (this time the original start)
            for bone in self.bones: # iterate forward this time (see above)
                bone.update_origin(current_pos)
                direction = normalize(np.array(bone.end_point) - current_pos)
                new_end = current_pos + direction * bone.length
                bone.update_endpoint(new_end)
                current_pos = new_end

            # Check if we're close enough to target
            if np.linalg.norm(self.bones[-1].end_point - target) < margin:  # find the distance vector between the tip
                                                                            # of the armature and the target, then get
                                                                            # its norm (magnitude)
                break


    @staticmethod
    def generate_random_armature(vwindow, origin: tuple, length: float, number: int):
        """Generates a random armature starting at the point 'origin' with 'number' segments each of length 'length'."""
        vector_bones = []
        last_bone_end = origin
        for i in range(number):
            rand_vec = random_vector(length)
            vector_bones.append(VectorBone(vwindow, last_bone_end, rand_vec))
            last_bone_end += rand_vec
        return Armature(vwindow, vector_bones)



class VectorWindow(pyglet.window.Window):
    """A basic simulation window for testing armatures."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_batch = pyglet.graphics.Batch()
        self.mouse_position = (0,0)
        self.armature = Armature.generate_random_armature(self, self.mouse_position, 150, 2)



    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self.mouse_position = (x,y)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == 1:
            self.armature = Armature.generate_random_armature(self, self.mouse_position, 150, 3)

    def on_draw(self):
        self.clear()
        self.line_batch.draw()

    def update(self, dt: float):

        self.armature.fabrik(self.mouse_position, 1)

        pass

if __name__ == '__main__':
    window = VectorWindow(width=1280, height=720, caption="Vector System")

    pyglet.clock.schedule_interval(window.update, 1 / 144)
    pyglet.app.run()