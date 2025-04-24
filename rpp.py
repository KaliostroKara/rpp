import numpy as np
import time
import multiprocessing
from mpi4py import MPI

# --- Генерація випадкових матриць ---
def generate_matrices(n):
    A = np.random.randint(0, 10, (n, n))
    B = np.random.randint(0, 10, (n, n))
    return A, B

# --- Послідовне множення ---
def sequential_multiply(A, B):
    return np.dot(A, B)

# --- Паралельне множення (OpenMP-подібне через multiprocessing) ---
def parallel_worker(args):
    A_row, B = args
    return np.dot(A_row, B)

def openmp_multiply(A, B):
    with multiprocessing.Pool() as pool:
        result = pool.map(parallel_worker, [(row, B) for row in A])
    return np.array(result)

# --- Паралельне множення (MPI) ---
def mpi_multiply(A, B, n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rows_per_proc = n // size
    local_A = np.zeros((rows_per_proc, n), dtype='i')
    B_global = np.zeros((n, n), dtype='i')
    C = np.zeros((n, n), dtype='i') if rank == 0 else None

    if rank == 0:
        comm.Bcast(B, root=0)
        comm.Scatter([A, MPI.INT], [local_A, MPI.INT], root=0)
    else:
        comm.Bcast(B_global, root=0)
        B = B_global
        comm.Scatter([None, MPI.INT], [local_A, MPI.INT], root=0)

    local_C = np.dot(local_A, B)

    comm.Gather(local_C, C, root=0)
    return C

# --- Основна програма ---
def main():
    n = 4  # Розмір матриці
    print("Оберіть реалізацію:")
    print("1 - Послідовна")
    print("2 - Паралельна (OpenMP / multiprocessing)")
    print("3 - Паралельна (MPI)")
    choice = input("Ваш вибір: ")

    if choice not in ['1', '2', '3']:
        print("Невірний вибір!")
        return

    if choice == '3':
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            A, B = generate_matrices(n)
        else:
            A = None
            B = None
        start = MPI.Wtime()
        C = mpi_multiply(A, B, n)
        end = MPI.Wtime()
        if rank == 0:
            print(f"[MPI] Час виконання: {end - start:.4f} секунд")
            print("\nМатриця A:\n", A)
            print("\nМатриця B:\n", B)
            print("\nРезультат множення (C = A * B):\n", C)
    else:
        A, B = generate_matrices(n)
        start = time.time()
        if choice == '1':
            C = sequential_multiply(A, B)
        elif choice == '2':
            C = openmp_multiply(A, B)
        end = time.time()
        method = "Послідовна" if choice == '1' else "OpenMP (multiprocessing)"
        print(f"[{method}] Час виконання: {end - start:.4f} секунд")
        print("\nМатриця A:\n", A)
        print("\nМатриця B:\n", B)
        print("\nРезультат множення (C = A * B):\n", C)

if __name__ == "__main__":
    main()
