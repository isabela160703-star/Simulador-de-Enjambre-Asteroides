import time
import multiprocessing as mp

from state import GameState
from physics import update_state
from parallel import update_state_parallel
from renderer import Renderer


def create_state(n, w, h):
    import numpy as np
    import random

    positions = np.array([
        [random.uniform(0, w), random.uniform(0, h)]
        for _ in range(n)
    ], dtype=float)

    velocities = np.zeros((n, 2))
    masses = np.ones((n, 1))

    return GameState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        width=w,
        height=h,
        num_asteroids=n
    )


def main():
    WIDTH, HEIGHT = 800, 600
    N = 300
    DT = 0.1

    state = create_state(N, WIDTH, HEIGHT)
    renderer = Renderer(WIDTH, HEIGHT)

    print("Simulación iniciada")

    while True:
        renderer.handle_events()

        start = time.time()

        state = update_state(state, DT)  # secuencial (cambia aquí si quieres paralelo)

        end = time.time()

        fps = 1 / max(end - start, 0.0001)

        renderer.draw(state, fps)
        renderer.clock.tick(60)


if __name__ == "__main__":
    main()