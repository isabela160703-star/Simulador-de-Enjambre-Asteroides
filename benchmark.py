import time
import multiprocessing as mp

from state import GameState
from physics import update_state
from parallel import update_state_parallel


def create_state(n, w, h):
    import numpy as np
    import random

    positions = np.array([
        [random.uniform(0, w), random.uniform(0, h)]
        for _ in range(n)
    ])

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


def run():
    WIDTH, HEIGHT = 800, 600
    STEPS = 50
    workers = mp.cpu_count()

    print("CPU cores:", workers)

    # SEC
    state = create_state(1000, WIDTH, HEIGHT)

    start = time.time()
    for _ in range(STEPS):
        state = update_state(state, 0.1)
    sec = time.time() - start

    print("Secuencial:", sec)

    # PAR
    state = create_state(1000, WIDTH, HEIGHT)
    pool = mp.Pool(workers)

    start = time.time()
    for _ in range(STEPS):
        state = update_state_parallel(state, 0.1, pool, workers)
    par = time.time() - start

    pool.close()
    pool.join()

    print("Paralelo:", par)

    print("Speedup:", sec / par)


if __name__ == "__main__":
    run()