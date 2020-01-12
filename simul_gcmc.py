#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sat Dec 22 11:24:53 2018
Modified by Ziyi Peng
@author: yaozhang
"""

################################################################################

import numpy as np
import math
import pandas as pd

np.set_printoptions(threshold = 1e6)  # threshold: 表示输出数组的元素数目
# translation: represents the number of elements in the output array

# Square box is used
grid_size = 28  # the length of the simulation lattice
n_steps = 20000  # total steps of gcmc
n_intervals = 40

TEMP = 298.0  # temperature, in Kelvin
Y_COEFF = 2  # the coefficient between water and csh solid
TEMP_C = 647.0  # critical temperature, in Kelvin

K_BOLT = 0.0019872041  # boltsman constant, in kcal/(mol?K)
MU_SAT = -2.0 * K_BOLT * TEMP_C  # saturation chemical potential
N_COORD = 4.0  # coordination number: square is 4, triangle is 3, in kcal/(mol?K)

# Interaction energy
W_FF = -2.0 * MU_SAT/N_COORD  # water-water interaction
W_MF = Y_COEFF * W_FF  # water-matrix interaction

################################################################################

# Determine the location of each box
n_squares = grid_size * grid_size  # total number of boxes in the lattice
print(n_squares + 1)
state = np.zeros((n_squares + 1,5)) # define the matrix to record the location and the state of each box
# the first column (state[:,1]) is x location, the second column is the y location, 
# the third is to define that the box is solid (0) or pore (1) and ... ???

# x location of each box
for i in range(1, n_squares + 1):
    state[i,1] = (i-1) % grid_size
# y location of each box
for i in range(1, n_squares + 1):
    state[i,2] = (i-1) // grid_size

################################################################################

# Define whether each square is a solid or pore box    
for i in range(1, n_squares + 1):
    if state[i,2] <= 12 and state[i,2] % 2 != 0:
        state[i,3] = 1
    elif state[i,2] >= 16 and state[i,2] % 2 == 0:
         state[i,3] = 1
    if (state[i,2] == 4) and (state[i,1] == 11 or state[i,1] == 16):
        state[i,3] = 1
    elif state[i,2] == 6 and (state[i,1] > 9 and state[i,1] < 18):
        state[i,3] = 1
    elif state[i,2] == 8 and (state[i,1] > 7 and state[i,1] < 20):
        state[i,3] = 1
    elif state[i,2] == 10 and (state[i,1] > 4 and state[i,1] < 23):
        state[i,3] = 1
    elif state[i,2] == 17 and (state[i,1] > 4 and state[i,1] < 23):
        state[i,3] = 1
    elif state[i,2] == 19 and (state[i,1] > 7 and state[i,1] < 20):
        state[i,3] = 1
    elif state[i,2] == 21 and (state[i,1] > 9 and state[i,1] < 18):
        state[i,3] = 1
    elif (state[i,2] == 23) and (state[i,1] == 11 or state[i,1] == 16):
        state[i,3] = 1
    if state[i,1] > 11 and state[i,1] < 16:
        state[i,3] = 1
    if state[i,2] > 11 and state[i,2] < 16:
        state[i,3] = 1

################################################################################

# Record the neighboring list
R_CUTOFF = 1.01  # cutoff radius
n_neighbors = np.zeros((n_squares + 1, 1), dtype=np.int)  # the number of neighboring number of each box
neighbors = np.zeros((n_squares + 1, n_squares), dtype=np.int)  # the matrix of neighboring list

for i in range(1, n_squares):
    for j in range(i+1, n_squares + 1): 
        dx = state[j,1]-state[i,1]
        dy = state[j,2]-state[i,2]
        dx = dx-round(dx/grid_size)*grid_size   # periodic boundary for x
        dy = dy-round(dy/grid_size)*grid_size   # periodic boundary for y
        d_squared = dx*dx+dy*dy

        if d_squared <= (R_CUTOFF ** 2):
            n_neighbors[i] += 1
            neighbors[i,int(n_neighbors[i])] = j
            n_neighbors[j] += 1
            neighbors[j,int(n_neighbors[j])] = i

max_neighbors = n_neighbors[:,0].max()           
neighbors = neighbors[:,0:max_neighbors+1]

################################################################################

# Calculate the energy of the system
def energy(neighbors, state): 
    H_W = 0
    H_WS = 0
    for i in range(1, n_squares + 1):
        for j in range(1, max_neighbors + 1):
            n = int(neighbors[i,j])
            H_W -= W_FF * state[i,4]*state[n,4] * state[i,3]*state[n,3]
            H_WS -= W_MF * state[i,4] * state[i,3]*(1-state[n,3])
    return H_W*0.5+H_WS

################################################################################

# Fill all pores
for i in range(1, n_squares + 1):
    if state[i,3] == 1:
        state[i,4] = 1
#print(state[1:51,4])

#calculate the ground energy       
H_T = energy(neighbors,state)
print(H_T)

# Reset the state of the pores
for i in range(1, n_squares + 1):
    state[i,4] = 0

################################################################################

H = np.zeros((n_steps+2,n_intervals))
rH = np.zeros((n_steps+2,n_intervals))

# Number of total pores
n_pores= int(sum(state[:,3]))
print(n_pores)
index = 1

# Record the order of pores in the matrix state
pores = np.zeros((n_pores+1, 1))
for i in range(1, n_squares + 1):
    if state[i,3] == 1:
        pores[index,0] = i
        index += 1

################################################################################

# GCMC Simulation

density = np.zeros((n_steps+2,n_intervals)) # define the matrix to record the density 
# (not the real density, but how many pores are filled with water)

for j in range(1,n_intervals):
    # random numbers determining which pore will be filled
    d = n_pores*np.random.rand(n_steps+2,1)  
    # random numbers determining whether the status change of the pore should be accepted
    p = np.random.rand(n_steps+2,1)  

    if j <= 20:
        RH=j*0.05  # increase relative humidity
    else:
        RH=1-(j-20)*0.05  # decrease relative humidity 

    mu = MU_SAT+K_BOLT*TEMP*math.log(RH)

    for i in range(1,n_steps+1):
        e = int(round(d[i,0]))  # chose the pore to be filled
        if e == 0:
            e = 1
        E = int(pores[e,0])

        n_full = state[:,4].sum(axis=0)  # calculate the number of full pores
        H0 = energy(neighbors,state)  # initial energy
        
        new_state = np.copy(state)
        new_state[E,4] = 1.0-new_state[E,4]  # change the pore status
        n_full_new = new_state[:,4].sum(axis=0)  # the number of filled pores 

        dN = n_full_new-n_full  # the change in the number of filled pores after each step
        H1 = energy(neighbors,new_state)
        
        # Energy change
        dH = H1-H0 
        du = dH-mu*dN
        
        f = math.exp(-du/K_BOLT/TEMP) # calculate the acceptance probability
        if du<0:
            state[E,4] = new_state[E,4]
        elif du>=0 and p[i,0]< f:
            state[E,4] = new_state[E,4]

        H[i,j] = energy(neighbors,state)
        rH[i,j] = H[i,j]/H_T
        n_full_new = sum(state[:,4])
        # print(n_full_new)
        density[i,j]= n_full_new/n_pores

################################################################################

# Calculate the average density  using the last 2000 steps       
average_density = np.zeros((n_intervals,1))
for jj in range(1,n_intervals):
    average_density[jj,0]=sum(density[(n_steps-2000):n_steps,jj])/2001

# Store the data   
energycshgel=pd.DataFrame(H)
energycshgel.to_csv('energycshgel.csv')
average_densitycshgel=pd.DataFrame(average_density)
average_densitycshgel.to_csv('average_densitycshgel.csv')












    
