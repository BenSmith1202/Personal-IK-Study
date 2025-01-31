import numpy
import numpy as np
import pyglet
from pyglet import shapes
from numpy import linalg


# lst = [10,20,30,40,50]
#
# vctr = np.array(lst)
#
# vctr = np.array(lst)
#
# print("Vector created from a list:")
# print(vctr)
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

class vector_bone:
    def __init__(self, vwindow, origin, vector: np.array, length=-1):
        self.vwindow = vwindow
        self.origin = np.array(origin)
        self.vector = np.array(vector)
        if length == -1:
            self.length = linalg.norm(vector)  # Calculate the length from the vector
        else:
            self.length = length
            self.vector = normalize(self.vector) * length  # Normalize and scale to the specified length

        self.update_endpoint(self.origin + self.vector)

    @staticmethod
    def from_coords(vwindow, origin: tuple, end_point: tuple, length=-1):
        origin = np.array(origin)
        end_point = np.array(end_point)
        vector = end_point - origin
        return vector_bone(vwindow, origin, vector, length)

    def update_endpoint(self, end_point):
        self.vector = end_point - self.origin
        self.vector = normalize(self.vector) * self.length  # Lock the length
        self.end_point = self.origin + self.vector  # Update the endpoint

        # Redraw the line
        self.line = pyglet.shapes.Line(
            self.origin[0], self.origin[1],
            self.end_point[0], self.end_point[1],
            2, batch=self.vwindow.line_batch
        )



class VectorWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_batch = pyglet.graphics.Batch()
        self.mouse_position = (0,0)
        self.click_pos = (0, 0)
        self.mouse_vector = vector_bone.from_coords(self, (0,0), self.mouse_position)
        self.extra_vector = vector_bone(self, (0, 0), numpy.array([0, 0]))



    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self.mouse_position = (x,y)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == 1:
            self.click_pos = (x,y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.mouse_vector = vector_bone.from_coords(self, self.click_pos, self.mouse_position, -1)
        pass

    def on_draw(self):
        self.clear()
        self.line_batch.draw()

    def update(self, dt: float):
        #   composite vector view
        #total_vector =  vector_bone.from_coords(self, self.click_pos, self.mouse_position)
        #self.mouse_vector = vector_bone.from_coords(self, self.click_pos, numpy.array((self.mouse_position[0], self.click_pos[1])))
        #self.extra_vector = vector_bone.from_coords(self, self.mouse_vector.end_point, numpy.array([self.mouse_vector.end_point[0], self.mouse_position[1]]))

        #
        self.mouse_vector.update_endpoint(self.mouse_position)
        self.extra_vector = vector_bone.from_coords(self, self.mouse_vector.end_point, self.mouse_vector.end_point + (0,-50), -1)
        pass

if __name__ == '__main__':
    window = VectorWindow(width=1280, height=720, caption="Vector System")

    pyglet.clock.schedule_interval(window.update, 1 / 144)
    pyglet.app.run()