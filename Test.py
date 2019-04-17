from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
# print("X0= {}",X0)
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1
# print("X1= {}",X1)
X = np.concatenate((X0.T, X1.T), axis = 1) # all data
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels
# print("X=")
# print(X)
# print("y=")
# print(y)
from cvxopt import matrix, solvers
# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)
# print("V=",V,"\n K=",K)
p = matrix(-np.ones((2*N, 1))) # all-one vector
# build A, b, G, h
G = matrix(-np.eye(2*N)) # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1)))
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)
# print("A=",A,"b=",b,"G=",G,"h=",h)
l = np.array(sol['x'])
print('lambda = ')
print(l)

epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]
print(S)
VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
print(V)
print(VS)
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)