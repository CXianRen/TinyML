import numpy as np


# read arguments from command line
import argparse
parser = argparse.ArgumentParser(description='Test matmul_ext')
parser.add_argument('--m', type=int, nargs='+', default=[2, 3],
                    help='List of integers for matrix m, e.g., --m 1 2 3')
parser.add_argument('--n', type=int, nargs='+', default=[3, 4],
                    help='List of integers for matrix n, e.g., --n 4 5 6')
parser.add_argument('--dtype', type=str, default='float32',
                    help='Data type of the matrices')
parser.add_argument('--save_path', type=str, default='',
                    help='Path to save the resulting matrix, if needed')

def main():
    args = parser.parse_args()
    m = args.m
    n = args.n
    dtype = args.dtype
    # set seed for reproducibility
    np.random.seed(42)
    # Create random matrices
    a = np.random.rand(*m).astype(dtype)
    b = np.random.rand(*n).astype(dtype)
    print(f"Matrix A shape: {a.shape}, dtype: {a.dtype}")
    print(f"Matrix B shape: {b.shape}, dtype: {b.dtype}")
    # perform dot product
    c = np.dot(a, b)
    # print("a :", a)
    # print("b :", b)
    # print("c :", c)
    print(f"Resulting matrix C shape: {c.shape}, dtype: {c.dtype}")
    if args.save_path:
        a.tofile(f"{args.save_path}/a.bin")
        b.tofile(f"{args.save_path}/b.bin")
        c.tofile(f"{args.save_path}/c.bin")

if __name__ == "__main__":
    main()