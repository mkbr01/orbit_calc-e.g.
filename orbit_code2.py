import numpy as np
import matplotlib.pyplot as plt

def rvect1(a, e, nu):
    return a * (1 - e ** 2) / (1 + e * np.cos(nu))

def orbitVelocity(r, MU, a):
    return np.sqrt(MU * ((2 / r) - (1 / a)))

# Globals
G = 6.673e-11
M_e = 5.972e24
r_e = 6.3781e6
MU = 398601
nu = []
nu1 = 0.174533
e_1 = 0.5

# Circular orbit velocity
V = orbitVelocity(G, M_e, 600000, r_e)

V_1 = np.sqrt(MU * ((2 / r) - (1 / a)))

vP = np.sqrt(G * M_e * (1 + e / P))

vA = np.sqrt(G * M_e * (1 - e / P))

# Radius vector function
r = rvect1(8000, 0.5285, 1.5708)

plt.figure(1)
plt.polar(r, 'o')
plt.ylim([0, 15000])

# Change in semi-major axis
r = rvect1(36000, 0.2, 1.5708)
plt.polar(r, 'o')

# Change in eccentricity
r = rvect1(36000, 0.895, 1.5708)
plt.polar(r, '*')

# Change in true anomaly
r = rvect1(36000, 0.2, 0.174533)
plt.polar(r, 'x')

plt.show()
