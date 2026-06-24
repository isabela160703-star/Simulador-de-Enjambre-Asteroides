import numpy as np

G = 0.5  
SOFTENING = 5.0  

def compute_forces(positions, masses):
    n = len(positions)
    forces = np.zeros((n, 2))

    for i in range(n):
        diff = positions - positions[i]
        dist_sq = np.sum(diff ** 2, axis=1) + SOFTENING ** 2

        force_mag = G * masses[i] * masses[:, 0] / dist_sq
        dist = np.sqrt(dist_sq)

        direction = diff / dist[:, None]
        forces[i] = (direction * force_mag[:, None]).sum(axis=0)

    return forces

def update_state(state, dt):
    forces = compute_forces(state.positions, state.masses)

    accelerations = forces / state.masses

    new_velocities = state.velocities + accelerations * dt
    new_positions = state.positions + new_velocities * dt

    # rebote
    new_velocities[new_positions[:, 0] < 0, 0] *= -1
    new_velocities[new_positions[:, 0] > state.width, 0] *= -1
    new_velocities[new_positions[:, 1] < 0, 1] *= -1
    new_velocities[new_positions[:, 1] > state.height, 1] *= -1

    new_positions = np.clip(
        new_positions,
        [0, 0],
        [state.width, state.height]
    )

    from state import GameState

    return GameState(
        positions=new_positions,
        velocities=new_velocities,
        masses=state.masses,
        width=state.width,
        height=state.height,
        num_asteroids=state.num_asteroids
    )