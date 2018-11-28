import numpy as np

# male, female, tall, short
ar = np.array([
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 1],
])

u, s, vh = np.linalg.svd(ar)

print(np.array2string(u, max_line_width=300, precision=3, suppress_small=True))
print(np.array2string(s, max_line_width=300, precision=3, suppress_small=True))
print(np.array2string(vh, max_line_width=300, precision=3, suppress_small=True))

aa = np.array([[1], [0], [1], [0]])
print()

print(np.array2string(u.transpose() @ ar, max_line_width=300, precision=3, suppress_small=True))
print(np.array2string(s @ vh, max_line_width=300, precision=3, suppress_small=True))

