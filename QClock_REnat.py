# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 21:46:44 2018

@author: Ренат
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.integrate as integrate


L = 50.
Nx = 1000
x,dx = np.linspace(-L, L, Nx,retstep = True)
T = 300.
Nt = 3000
t,dt = np.linspace(0, T, Nt,retstep = True)
center = 3. #Положение центра ямы
width = 3. #Ширина ямы


def normal(psi, x):
    return integrate.simps(np.abs(psi)**2, x)

def Psi_init(x,x0,sigma):
    return np.exp(-(x+x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi*sigma**2))

def Potential(x,center,width):
    U0 = 0.65
    V = np.zeros(x.size, dtype=np.complex64)
    V[(x >= -center - 0.5 * width) * (x <= -center + 0.5 * width)] = -U0
    V[(x >= center - 0.5 * width) * (x <= center + 0.5 * width)] = -U0
    return V

def Shred_Solve(Nt, dt, dx, V, psi0,x):
    psi = np.zeros((psi0.size, Nt), dtype=np.complex128)
    psi[:, 0] = psi0
    diff = dt / dx**2
    psi[1:-1, 1] = psi[1:-1, 0] + 1j * diff * (psi[2:, 0] - 2. * psi[1:-1, 0] + psi[:-2, 0]) - 1j * dt * V[1:-1] * psi[1:-1, 0]
    norm= normal(psi[:,1],x)
    psi[:,1] /= np.sqrt(norm)
    for i in range(2, Nt):
        psi[1:-1, i] = (1/dx**2*(psi[2:,i-1]+psi[:-2,i-1])-psi[1:-1,i-2]*(1/dx**2+V[1:-1]+1j/dt))/(1/dx**2+V[1:-1]-1j/dt)
        norm= normal(psi[:,i],x)
        psi[:,i] /= np.sqrt(norm)

    return psi


psi0 = Psi_init(x,center,width/2)
V = Potential(x,center,width)
plt.plot(x, np.abs(psi0)**2)
plt.plot(x,V)

psi = Shred_Solve(Nt,dt,dx,V,psi0,x)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_xlabel('Time')
ax.set_ylabel('Coordinates')
ax.pcolormesh(t, x, np.real(np.abs(psi)**2))
plt.show()

mult = round(0.3 / dt)
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(x, np.abs(psi[:,0])**2)

def animate(i):
    line.set_ydata(np.abs(psi[:,int(i)])**2)
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1,(t.size) / mult), interval = 1)
ani.save('Renat_QClock.html')
