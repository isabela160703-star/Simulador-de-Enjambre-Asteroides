import numpy as np
import random


class SimulationState:
    def __init__(self, num_asteroids, width, height):
        self.width = width
        self.height = height

        # posiciones (x, y)
        self.positions = np.array([
            [random.uniform(0, width), random.uniform(0, height)]
            for _ in range(num_asteroids)
        ], dtype=float)

        # velocidades (vx, vy)
        self.velocities = np.zeros((num_asteroids, 2), dtype=float)

        # masas
        self.masses = np.ones((num_asteroids, 1), dtype=float)
