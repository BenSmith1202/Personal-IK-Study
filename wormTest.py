import pyglet
from pyglet import shapes
from pyglet.gl import *
from math import dist, sqrt, atan2, degrees, cos, sin, radians, pi
from pyglet.window import key
from time import time

class WormSegment:
    def __init__(self, wwindow, x, y, angle, radius):
        self.x = x
        self.y = y
        self.angle = angle
        self.circle = shapes.Circle(x, y, radius, color=(255,255,255,255), batch=wwindow.fill_batch)


    def change_pos(self, dx, dy):
        self.x += dx
        self.circle.x = self.x
        self.y += dy
        self.circle.y = self.y



def calculate_angle(current_x, current_y, target_x, target_y):
    dx = target_x - current_x
    dy = target_y - current_y
    angle = atan2(dy, dx)
    #angle = degrees(angle)
    return angle


class PACreature:
    def __init__(self, x: float, y: float, speed: float, wwindow, num_segments: int, segment_distance: float):
        self.x: float = x
        self.y: float = y
        self.speed: float = speed
        self.wwindow = wwindow
        self.num_segments = num_segments
        self.segment_distance = segment_distance
        self.spine = []

    def update(self, target):
        pass


class Worm(PACreature):
    def __init__(self, x, y, speed, wwindow, num_segments, segment_distance, head_radius, max_angle):
        super().__init__(x, y, speed, wwindow, num_segments, segment_distance)
        for i in range(0, num_segments):
            self.spine.append(WormSegment( self.wwindow, x - (i * segment_distance), y, 0, max(1, head_radius * (1 - 0.3 * i / num_segments))))
        self.max_angle = max_angle
    def update(self, target):
        current_target = target
        for i, segment in enumerate(self.spine):
            target_distance = dist((segment.x, segment.y), current_target)
            if target_distance > self.segment_distance:
                if i == 0:
                    # this is the head
                    desired_angle = calculate_angle(segment.x, segment.y, current_target[0], current_target[1])
                    current_angle = segment.angle
                    delta_angle = (desired_angle - current_angle + pi) % (2 * pi) - pi
                    if delta_angle > radians(self.max_angle):
                        delta_angle = radians(self.max_angle)
                    if delta_angle < -radians(self.max_angle):
                        delta_angle = -radians(self.max_angle)
                    segment.angle += delta_angle
                else:
                    segment.angle = calculate_angle(segment.x, segment.y, current_target[0], current_target[1])
                distance = min(target_distance - self.segment_distance, self.speed)
                segment.change_pos(cos(segment.angle)*distance, sin(segment.angle)*distance)
            current_target = (segment.x, segment.y)



class WormWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fill_batch = pyglet.graphics.Batch()
        self.line_batch = pyglet.graphics.Batch()
        self.mouse_position = (0,0)
        self.worm1 = Worm(self.width/2, self.height/2, 5, self, 100, 10, 15, 5)
        self.cursor = shapes.Circle(0, 0, 5, color=(255,255,255,255), batch=self.fill_batch)


    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self.mouse_position = (x,y)

    def on_draw(self):
        self.clear()
        self.fill_batch.draw()
        self.line_batch.draw()

    def update(self, dt: float):  # physics update. run rate controlled in main
        self.cursor.x = self.mouse_position[0]
        self.cursor.y = self.mouse_position[1]
        self.worm1.update(self.mouse_position)
        pass


if __name__ == '__main__':
    window = WormWindow(width=1280, height=720, caption="Planetary System")
    window.set_mouse_visible(False)
    pyglet.clock.schedule_interval(window.update, 1 / 60)
    pyglet.app.run()


