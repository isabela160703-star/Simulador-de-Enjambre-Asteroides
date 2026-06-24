import numpy as np
import multiprocessing as mp
from physics import G, SOFTENING

def worker_compute_forces(args):
    positions, masses, start_idx, end_idx = args
    chunk_forces = np.zeros((end_idx - start_idx, 2))
    
    for idx, i in enumerate(range(start_idx, end_idx)):
        diff = positions - positions[i]
        dist_sq = np.sum(diff*2, axis=1) + SOFTENING*2
        force_mag = G * masses[i] * masses[:, 0] / dist_sq
        
        dist = np.sqrt(dist_sq)
        direction = diff / dist[:, np.newaxis]
        chunk_forces[idx] = np.sum(direction * force_mag[:, np.newaxis], axis=0)
    return chunk_forces

def update_state_parallel(state, dt, pool, num_workers):
    n = state.num_asteroids
    chunk_size = n // num_workers
    tasks = []
    
    for i in range(num_workers):
        start = i * chunk_size
        end = n if i == num_workers - 1 else (i + 1) * chunk_size
        tasks.append((state.positions, state.masses, start, end))

    results = pool.map(worker_compute_forces, tasks)
    forces = np.vstack(results)

    accelerations = forces / state.masses
    state.velocities += accelerations * dt
    state.positions += state.velocities * dt

    state.velocities[state.positions[:, 0] < 0, 0] *= -1
    state.velocities[state.positions[:, 0] > state.width, 0] *= -1
    state.velocities[state.positions[:, 1] < 0, 1] *= -1
    state.velocities[state.positions[:, 1] > state.height, 1] *= -1
    state.positions = np.clip(state.positions, [0, 0], [state.width, state.height])