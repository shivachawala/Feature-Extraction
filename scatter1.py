import sys
import numpy as np

if len(sys.argv) != 5:
    print('usage:', sys.argv[0], 'data_file label_file')
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

# in all values of X
# we already known that values are seperated as 3 groups:1,2,3
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

s1 = np.zeros((n, n))
s2 = np.zeros((n, n))
s3 = np.zeros((n, n))

for i, x in enumerate(X):
    if y[i] == 1:
        s1 += np.dot((x-u1).T, x-u1)
    if y[i] == 2:
        s1 += np.dot((x-u2).T, x-u2)
    if y[i] == 3:
        s1 += np.dot((x-u3).T, x-u3)

W = s1 + s2 + s3

eig_vals, eig_vecs = np.linalg.eigh(W)
print("evals=", eig_vals, " evecs=", eig_vecs)

id = np. argsort(eig_vals)[:: 1]
eig_vals = eig_vals[id]
eig_vecs = eig_vecs[:, id]

# reduce the dimensionality of the data to 2
r = 2
eig_vecs_2 = eig_vecs[:, :r]
eig_vec_T = eig_vecs_2.T
np.savetxt(sys.argv[3], eig_vec_T, delimiter=',')

res = np.dot(X, eig_vecs_2)
np.savetxt(sys.argv[4], res, delimiter=',')
