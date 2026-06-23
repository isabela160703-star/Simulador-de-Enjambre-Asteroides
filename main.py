import time
from state import SimulationState
from renderer import Renderer
from physics import update_state

def main():
    WIDTH, HEIGHT = 800, 600
    NUM_ASTEROIDS = 150 # Puedes subir esto, pero el FPS bajará en modo serial
    DT = 0.2

    state = SimulationState(NUM_ASTEROIDS, WIDTH, HEIGHT)
    renderer = Renderer(WIDTH, HEIGHT)

    print("Iniciando simulación...")
    
    while True:
        renderer.handle_events()
        
        start_time = time.time()
        # Calcula la física del cuadro actual
        update_state(state, DT)
        end_time = time.time()
        
        # Prevenir división por 0
        compute_time = max(end_time - start_time, 0.001)
        fps = 1.0 / compute_time
        
        renderer.draw(state, fps)
        renderer.clock.tick(60) # Límite de 60 cuadros por segundo

if __name__ == "__main__":
    main()