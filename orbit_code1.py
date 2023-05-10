import numpy as np
import matplotlib.pyplot as plt

def rvect1(a, e_1, nu):
    return ((a * (1 - (e_1 ** 2))) / ((1 + e_1) * np.sin(nu)))

def orbitVelocity(r, MU, a):
    return np.sqrt((MU * ((2 / r) - (1 / a))))

G = 6.673e-11  # Gravitational constant
M = 5.98e24  # mass of earth in (kg)
ra = 100000  # orbit distance in (m)
r = 6.37e6 + ra  # total radius of orbit in (m)
m = 500  # mass satellite (kg)
a = (G * M) / (r ** 2)  # check for acceleration
v_orb = np.sqrt((G * M) / r)  # orbital velocity (m/s)
T = np.sqrt(((4 * (np.pi ** 2)) * r ** 3) / (G * M))  # period (s)

simt = T
t = np.arange(1, simt+1, 150)
v_o = np.sqrt((G * M) / r) * np.ones_like(t)
rn = 6.37e6 + ra * np.ones_like(t)

plt.figure()
plt.plot(t, v_o)
plt.xlabel('Simulation Time', fontsize=12)
plt.ylabel('Velocity', fontsize=12)
plt.grid()

plt.figure()
plt.plot(t, rn)
plt.xlabel('Simulation Time', fontsize=12)
plt.ylabel('Radius', fontsize=12)
plt.grid()


theta, rho = np.cart2pol(t, rn)
# th = np.arctan2(rn, t)
th = 2 * np.pi * t / T
rho = np.sqrt((t ** 2) + (rn ** 2))
theta = theta * (180 / np.pi)  # to degrees

# plt.figure()
# plt.polar(np.radians(theta), rho)
# plt.title('Orbit')
# plt.show()


G = 6.673e-11  # Gravitational constant
M = 5.98e24  # mass of earth in (kg)
ra = 100000  # orbit distance in (m)
r = 6.37e6 + ra  # total radius of orbit in (m)
m = 500  # mass satellite (kg)

MU = 24700
v_orb = np.sqrt((G * M) / r)  # circular orbital velocity (m/s)

a = (G * M) / (r ** 2)  # check for acceleration

# radius vector function
r = rvect1(24000, 0.5285, 1.507)

T = np.sqrt(((4 * (np.pi ** 2)) * r ** 3) / (G * M))  # period (s)

# elliptical orbital velocity function
V_1 = orbitVelocity(r, MU, a)

steps = T / 35
simt = -steps
t = np.zeros(36)
v_o = np.zeros(36)
rn = np.zeros(36)
for i in range(36
