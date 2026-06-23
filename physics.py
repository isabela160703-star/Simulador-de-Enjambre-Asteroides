import numpy as np

G = 0.5  
SOFTENING = 5.0  

def compute_forces(positions, masses):
    n = len(positions)
    forces = np.zeros((n, 2))
    for i in range(n):
        diff = positions - positions[i]
        dist_sq = np.sum(diff**2, axis=1) + SOFTENING**2
        force_mag = G * masses[i] * masses[:, 0] / dist_sq
        
        dist = np.sqrt(dist_sq)
        direction = diff / dist[:, np.newaxis]
        forces[i] = np.sum(direction * force_mag[:, np.newaxis], axis=0)
    return forces
