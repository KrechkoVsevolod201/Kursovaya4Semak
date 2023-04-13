import numpy as np
import math
import matplotlib.pyplot as plt
I = 100
K = 20000
l = 4
ka = 0.05
C = 1.4
betta=0.1
R=3
T = 150
hz = l / I
ht = T / K

z = np.linspace(0, l, I)
t = np.linspace(0, T, K)
w = np.zeros((I, K));
fz=300*betta*math.e**(-betta*z)
constant = ht*(ka/C)/(hz**2)

for k in range(K-1):
    i = 1
    while i<I-1:
        w[i, k+1] = constant*(w[i+1,k]-2*w[i,k]+w[i-1,k])+w[i,k]+ht*fz[i]
        i+=1
    w[0, k+1] = constant*(2*w[1,k]-2*w[0,k])+w[0,k]+ht*fz[0]
    w[i, k+1] = constant*(w[i-1,k]-w[i,k])+w[i,k]+ht*fz[i]

w_l = w[:, int(K/2)]
plt.plot(z, w_l)
w_l = w[:, int(K/5)]
plt.plot(z, w_l)
w_l = w[:, int(K/10)]
plt.plot(z, w_l)
w_l = w[:, int(K/15)]
plt.plot(z, w_l)
w_l = w[:, int(K/20)]
plt.plot(z, w_l)
plt.grid()
plt.show()