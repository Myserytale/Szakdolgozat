import numpy as np


def inspect_array(name, data):
    arr = np.asarray(data)

    print(f"\n{name}")
    print("Type:", type(data))
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)
    print("First values:", arr.flat[:20])

    if arr.size and (np.issubdtype(arr.dtype, np.integer) or arr.size < 100_000):
        values, counts = np.unique(arr, return_counts=True)
        print("Unique values:", list(zip(values[:30], counts[:30])))
