import random
import numpy as np

def generate_instance(n: int):
    random.seed(42)
    # U
    U:list = [random.randint(0,2)for _ in range(n)]
    # C
    C = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i, n, 1):
            C[i][j] = random.randint(0,1)
            C[j][i] = C[i][j]
            if i == j:
                C[i][j] = 0

    with open(f"pco_{n}.txt", "+w") as f:
        f.write("N\n")
        f.write(f"{n}\n")
        f.write("U\n")
        for i in range(n):
            f.write(f"{i};{U[i]}\n")
        
        f.write("C\n")
        for i in range(n):
            for j in range(n):
                f.write(f"{i},{j};{C[i][j]}\n")
    return 


if __name__ == '__main__':
    generate_instance(3)