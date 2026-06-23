import time
import multiprocessing as mp
from state import SimulationState
from physics import update_state
from parallel import update_state_parallel

def run_benchmark():
    NUM_ASTEROIDS = 600
    STEPS = 50
    WIDTH, HEIGHT = 800, 600
    
    print(f"--- Iniciando Benchmark ---")
    print(f"Asteroides: {NUM_ASTEROIDS} | Pasos de simulación: {STEPS}")
    
    # --- PRUEBA SERIAL ---
    state_serial = SimulationState(NUM_ASTEROIDS, WIDTH, HEIGHT)
    start_serial = time.time()
    for _ in range(STEPS):
        update_state(state_serial, 0.1)
    serial_time = time.time() - start_serial
    print(f"\nTiempo Serial (1 núcleo): {serial_time:.4f} segundos")

    # --- PRUEBA PARALELA ---
    state_parallel = SimulationState(NUM_ASTEROIDS, WIDTH, HEIGHT)
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    
    start_parallel = time.time()
    for _ in range(STEPS):
        update_state_parallel(state_parallel, 0.1, pool, num_workers)
    parallel_time = time.time() - start_parallel
    
    pool.close()
    pool.join()
    
    print(f"Tiempo Paralelo ({num_workers} núcleos): {parallel_time:.4f} segundos")
    
    # Resultados
    if parallel_time < serial_time:
        speedup = serial_time / parallel_time
        print(f"\nResultado: Paralelo fue {speedup:.2f}x veces más rápido.")
    else:
        print("\nResultado: Serial fue más rápido (sucede con pocos asteroides por el costo de crear los procesos).")

if __name__ == "__main__":
    # Necesario en Windows para multiprocessing
    mp.freeze_support()
    run_benchmark()