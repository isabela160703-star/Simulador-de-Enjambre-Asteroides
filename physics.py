import numpy as np
import multiprocessing as mp
from physics import G, SOFTENING

def worker_compute_forces(args):
    positions, masses, start_idx, end_idx = args
    chunk_forces = np.zeros((end_idx - start_idx, 2))

    for idx, i in enumerate(range(start_idx, end_idx)):
        diff = positions - positions[i]
        dist_sq = np.sum(diff**2, axis=1) + SOFTENING**2
        force_mag = G * masses[i] * masses[:, 0] / dist_sq

        dist = np.sqrt(dist_sq)
        direction = diff / dist[:, np.newaxis]
        chunk_forces[idx] = np.sum(direction * force_mag[:, np.newaxis], axis=0)

    return chunk_forces