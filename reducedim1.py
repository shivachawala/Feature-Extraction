import sys
import numpy as np

if len(sys.argv) != 5:
    # 'data_file labels_file output_file1'
    print('usage: ', sys.argv[0], 'data_file labels_file output_file1')
    sys.exit()

# get matrix from files
X = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True)
y = np.genfromtxt(sys.argv[2])
transpose = X.T
Result = np.dot(transpose, X)
eig_vals, eig_vecs = np.linalg.eig(Result)

id = np. argsort(eig_vals)[:: -1]
eig_vals = eig_vals[id]
eig_vecs = eig_vecs[:, id]
print(" evals =", eig_vals, "  evecs =", eig_vecs)

r = 2
Vec_r = eig_vecs[:, :r]
V = Vec_r.T
np.savetxt(sys.argv[3], V, delimiter=',')

res = np.dot(X, Vec_r)
np.savetxt(sys.argv[4], res, delimiter=',')
