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

def update_state(state, dt):
    forces = compute_forces(state.positions, state.masses)
    
    # F = M * A -> A = F / M
    accelerations = forces / state.masses
    state.velocities += accelerations * dt
    state.positions += state.velocities * dt

    # Rebote en los bordes de la pantalla
    state.velocities[state.positions[:, 0] < 0, 0] *= -1
    state.velocities[state.positions[:, 0] > state.width, 0] *= -1
    state.velocities[state.positions[:, 1] < 0, 1] *= -1
    state.velocities[state.positions[:, 1] > state.height, 1] *= -1
    state.positions = np.clip(state.positions, [0, 0], [state.width, state.height])