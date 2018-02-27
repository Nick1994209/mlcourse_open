import numpy as np

a = np.array((
    [
        [10, 12, 13],
        [0, 1, 2],
    ],
    [
        [100, 101, 102],
        [110, 112, 113],
    ],
    [
        [100, 101, 102],
        [110, 112, 113],
    ],
))
# print(a.shape)
#
# print(a.ravel())
# print(a.transpose())

# print(len(a[0]))
# print(len(a[0][0]))
# print(len(a[0]) * len(a[0][0]))
# print(a.reshape((len(a), len(a[0]) * len(a[0][0]))))
# print(a.reshape((len(a), -1)))
# print(a.reshape((a.shape[0], -1)))
#
# # print(a[[0, 1]])

print()

print(a.reshape(-1))

print(np.sort(a.reshape(-1)))
print(np.argsort(a.reshape(-1)))
print(a.reshape(-1)[np.argsort(a.reshape(-1))])

print(np.argmax(np.bincount([1, 1, 2, 5, 5, 5, 5000004])))
