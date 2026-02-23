import timeit
import random
import os
import numpy as np

# --- CONFIGURATION ---
ITERATIONS = 100_000
DIMENSIONS = 1536  # Standard OpenAI text-embedding-3-small dimension
NUM_BYTES = DIMENSIONS // 8 # 192 bytes

# --- DATA PREPARATION ---
# Generate two random hex strings representing two different embeddings
# os.urandom is the fastest way to get random bytes
hex_a = os.urandom(NUM_BYTES).hex()
hex_b = os.urandom(NUM_BYTES).hex()

print(f"Benchmark Configuration:")
print(f"- Dimensions: {DIMENSIONS} bits")
print(f"- Hex String Length: {len(hex_a)} chars")
print(f"- Iterations: {ITERATIONS:,}\n")

# --- METHOD 1: Table Lookup (The one you asked for) ---
# Pre-compute the table (0-255)
POPCOUNT_TABLE = [bin(i).count('1') for i in range(256)]
DISTANCE_TABLE = dict([(x+y, (int(x, 16)^int(y, 16)).bit_count()) for x in "0123456789abcdef" for y in "0123456789abcdef"])
DISTANCE_TABLE2 = [((i >> 4) ^ (0x0F & i)).bit_count() for i in range(256)]

def method_lookup(h1, h2):
    # 1. Conversion (Fast)
    b1 = bytes.fromhex(h1)
    b2 = bytes.fromhex(h2)
    
    # 2. Iteration (Slow in Python)
    distance = 0
    # zip creates an iterator, passing tuples to the loop
    for x, y in zip(b1, b2):
        distance += POPCOUNT_TABLE[x ^ y]
    return distance

def method_lookup2(h1, h2):
    distance = 0
    for x, y in zip(h1, h2):
        distance += DISTANCE_TABLE[x+y]
    return distance


# --- METHOD 2: Native Python (3.10+) ---
def method_native(h1, h2):
    # 1. Convert huge string to huge int (C-optimized)
    i1 = int(h1, 16)
    i2 = int(h2, 16)
    
    # 2. XOR and count (C-optimized)
    return (i1 ^ i2).bit_count()

# --- METHOD 3: Numpy ---
def method_numpy(h1, h2):
    # 1. Convert hex to bytes, then to numpy array
    a_bytes = np.frombuffer(bytes.fromhex(h1), dtype=np.uint8)
    b_bytes = np.frombuffer(bytes.fromhex(h2), dtype=np.uint8)
    
    # 2. Vectorized XOR and lookup via unpackbits (summing 1s)
    # Note: unpackbits is generally faster than iterating, but has overhead
    return np.unpackbits(np.bitwise_xor(a_bytes, b_bytes)).sum()

# --- RUNNING BENCHMARKS ---

def run_test(name, func):
    # Verify correctness first
    try:
        assert func(hex_a, hex_b) == method_native(hex_a, hex_b)
    except Exception as e:
        print(f"Skipping {name} due to error (maybe old Python version?): {e}")
        return

    print(f"Testing {name}...", end="", flush=True)
    
    # Run the timer
    total_time = timeit.timeit(lambda: func(hex_a, hex_b), number=ITERATIONS)
    avg_time_us = (total_time / ITERATIONS) * 1_000_000
    
    print(f" Done.")
    print(f"  -> Total: {total_time:.4f}s")
    print(f"  -> Per Call: {avg_time_us:.2f} microseconds")
    print("-" * 30)
    return total_time

if __name__ == "__main__":
    t_lookup2 = run_test("Table Lookup2 (Loop)", method_lookup2)
    t_lookup = run_test("Table Lookup (Loop)", method_lookup)
    t_native = run_test("Native Int (.bit_count)", method_native)
    t_numpy  = run_test("Numpy (Vectorized)", method_numpy)

    print("\n--- CONCLUSION ---")
    fastest = min(t_lookup, t_native, t_numpy)
    
    if fastest == t_native:
        print(f"WINNER: Native Python is {t_lookup/t_native:.1f}x faster than Lookup.")
    elif fastest == t_lookup:
        print(f"WINNER: Lookup Table is {t_native/t_lookup:.1f}x faster than Native.")
    else:
        print("WINNER: Numpy.")
