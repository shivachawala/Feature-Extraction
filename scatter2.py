import numpy as np
import sys

if len(sys.argv) != 5:
    print("usage:", sys.argv[0], "data_file labels_file")
    sys.exit()

# get matrix from file
X = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True)
y = np.genfromtxt(sys.argv[2])

# get the X is a m*n matrix
m, n = X.shape

a1 = np.zeros((1, n))
a2 = np.zeros((1, n))
a3 = np.zeros((1, n))

b1 = 0
b2 = 0
b3 = 0

for i, x in enumerate(X):
    if y[i] == 1:
        a1 += x
        b1 += 1
    if y[i] == 2:
        a2 += x
        b2 += 1
    if y[i] == 3:
        a3 += x
        b3 += 1

u1 = a1/b1
u2 = a2/b2
u3 = a3/b3


u = (a1+a2+a3)/(b1+b2+b3)

B = np.dot((u1-u).T, u1-u) * b1
B += np.dot((u2-u).T, u2-u) * b2
B += np.dot((u3-u).T, u3-u) * b3


eig_vals, eig_vecs = np.linalg.eigh(B)
print("evals=", eig_vals, " evecs=", eig_vecs)

id = np.argsort(eig_vals)[:: -1]  # sort in reverse order
eig_vals = eig_vals[id]
eig_vecs = eig_vecs[:, id]

r = 2
eig_vecs_2 = eig_vecs[:, :r]
eig_vec_T = eig_vecs_2.T
np.savetxt(sys.argv[3], eig_vec_T, delimiter=',')

res = np.dot(X, eig_vecs_2)
np.savetxt(sys.argv[4], res, delimiter=',')
