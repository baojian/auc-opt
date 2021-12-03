import numpy as np

for d in [5, 6, 7, 8, 9, 10]:
    print(f"d: [{d}]: ", end=" ")
    for alpha in [.75, .8, .85, .9, .95]:
        print(np.ceil(d * np.log(.1) / np.log(alpha) + np.log(.5) / np.log(alpha)), end=' ')
    print()
print()
for d in [5, 6, 7, 8, 9, 10]:
    print(f"d: [{d}]: ", end=" ")
    for alpha in [.75, .8, .85, .9, .95]:
        print(np.ceil(d * np.log(.1) / np.log(alpha)), end=' ')
    print()
print()
tau = .85
for d in [5, 6, 7, 8, 9, 10]:
    print(f"d: [{d}]: ", end=" ")
    for alpha in [.75, .8, .85, .9, .95]:
        print(np.ceil(d * np.log(.1) / np.log(alpha * tau) + np.log(.5) / np.log(alpha * tau)), end=' ')
    print()
print()
