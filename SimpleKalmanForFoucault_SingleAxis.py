#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:07:22 2022

@author: basile
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import scipy as sc
import scipy.linalg as sla


mu = 0.1             # Pixel size
nSamps = 1000        # number of samples
h = 0.05             # sample time (camera fps)

# Actual pendulum pulsation frequency and relative damping
omega = 2.0 * np.pi * 0.303;    # Pendulum pulsation
zeta = 0.0001;                  # Pendulum relative damping
x0 = [0.5, 0, 0.3]              # initial conditions [amplitude, speed, offset]

# Model (Kalman) pendulum pulsation frequency and relative damping
omegaM = 2.0 * np.pi * 0.3;     # Pendulum pulsation
zetaM = 0.0;                    # Pendulum relative damping
x0M = [0, 0, 0]                 # initial conditions [amplitude, speed, offset]



def discretePendulumModel(omega, zeta, h):
    # Continuous linear model on one axis, without offset
    # x'' + 2*omega*zeta* x' + omega^2 x = u
    # State space form with x1=x and x2=x' :
    Ac = np.asarray([[0, 1],[-omega**2, -2.0*omega*zeta]])
    Bc = np.asarray([0,1])
    
    # Discretize equations
    # https://en.wikipedia.org/wiki/Discretization
    Ad = sla.expm(h * Ac)
    Bd = np.linalg.inv(Ac) @ (Ad - np.eye(2)) @ Bc
    
    # Add an unknown constant state representing a measurement offset (camera not 
    # centered exactly)
    AAd = np.zeros((3,3))
    AAd[0:2,0:2] = Ad
    AAd[2,2] = 1.0 # offset remains constant
    BBd = np.zeros((3,1))
    BBd[0:2,0] = Bd 
    
    # Output is x + offset = x1 + offset
    # Output equation
    CCd = np.asarray([[1,0,2]]) # Since the state is [x, x', offset]
    
    return (AAd, BBd, CCd)

# Actual pendulum and pendulum model for Kalman observer
pendulum = discretePendulumModel(omega, zeta, h)
pendulumM = discretePendulumModel(omegaM, zetaM, h)



def propagateDiscreteModel(model, x0, inputVal):
    A = np.asarray(model[0])
    B = np.asarray(model[1])
    C = np.asarray(model[2])
    nSamples = inputVal.shape[1]
    nState = A.shape[0]
    nIn = B.shape[1]
    nOut = C.shape[0]
    # Propagate and plot some time evolution
    state = np.zeros((nState, nSamples))
    state[:,0] = x0# initial condition
    inputVal = np.zeros((nIn, nSamples))
    outputVal = np.zeros((nOut, nSamples))
    outputVal[:,0] = C @ state[:,0]
    for k in range(1,nSamples):
        state[:,k] = A @ state[:,k-1] + B @ inputVal[:,k]
        outputVal[:,k] = C @ state[:,k]
    return (state, outputVal)



# Kalman with notations from
# https://en.wikipedia.org/wiki/Kalman_filter

# covariance of the observation noise, we "estimate" it from the pixel size mu
Rk = (mu**3 / 12.0) * np.ones((1,1))
# covariance of the process noise, beuhhh
Qk = 1 * h * (mu**3 / 12.0) * np.diag([1, 1*omega, 1/100])


class FoucaultKalman():
    def __init__(self, model, Qk, Rk):
        self.Fk = np.asarray(model[0]) # state transition matrix (i.e. A)
        self.Bk = np.asarray(model[1]) # control input matrix (i.e. B)
        self.Hk = np.asarray(model[2]) # measurement model matrix (i.e. C)
        self.Qk = np.asarray(Qk) # covariance of the process noise (we assume it is zero)
        self.Rk = np.asarray(Rk) # covariance of the observation noise
        self.nState = self.Fk.shape[0]
        self.nIn = self.Bk.shape[1]
        self.nOut = self.Hk.shape[0]
        
    def predict(self, xk1k1, uk, Pk1k1):
        xkk1 = self.Fk @ xk1k1 + self.Bk @ uk
        Pkk1 = self.Fk @ Pk1k1 @ self.Fk.transpose() + self.Qk
        return (xkk1, Pkk1)
    
    def update(self, zk, xkk1, Pkk1):
        yk = zk - self.Hk @ xkk1
        Sk = self.Hk @ Pkk1 @ self.Hk.transpose() + self.Rk
        # Kalman gain
        Kk = Pkk1 @ self.Hk.transpose() @ np.linalg.inv(Sk)
        xkk = xkk1 + Kk @ yk
        Pkk = (np.eye(self.nState) - Kk @ self.Hk) @ Pkk1
        ykk = zk - self.Hk @ xkk1
        return (xkk, Pkk)
        
    def propagate(self, controlValues, measurementValues, measurementValuesAvailable):
        measurementValues = np.asarray(measurementValues)
        controlValues = np.asarray(controlValues)
        N = measurementValues.shape[1]
        stateEstimate = np.zeros((self.nState, N))
        
        xk1k1 = np.zeros((self.nState)) # Predicted (a priori) state estimate
        Pk1k1 = 0.000*np.eye(self.nState) # Predicted (a priori) estimate covariance 
        for k in range(0, N):
            uk = controlValues[:,k]
            if (measurementValuesAvailable[k]):
                # Measurement available
                zk = measurementValues[:,k]
                (xkk1, Pkk1) = self.predict(xk1k1, uk, Pk1k1)
                (xkk, Pkk) = self.update(zk, xkk1, Pkk1)
            else:
                # No measurement available
                (xkk1, Pkk1) = self.predict(xk1k1, uk, Pk1k1)
                (xkk, Pkk) = (xkk1, Pkk1)
                
            stateEstimate[:,k] = xkk # Keep state estimate k
            xk1k1 = xkk # for next step
            Pk1k1 = Pkk # for next step
        outputEstimate = self.Hk @ stateEstimate
        return (stateEstimate, outputEstimate)
        

    
    
        
kalman = FoucaultKalman(pendulumM, Qk, Rk)       

# Propagate (simulate) the pendulum
x0 = [1, 0,1.3]
inputVal = np.zeros([1, nSamps])
(pendulumState, pendulumMeasurement) = propagateDiscreteModel(pendulum, x0, inputVal)

# Run Kalman observer on simulated measurement
measurementValuesAvailable = np.full((nSamps), True) # Measurement always available
(stateEstimate, outputEstimate) = kalman.propagate(inputVal, pendulumMeasurement, measurementValuesAvailable)

plt.figure()   
plt.title('Mesurement always available')   
plt.plot(pendulumMeasurement.transpose(),'.', label='measurement') 
plt.plot(outputEstimate.transpose(),  label='output estimate')  

plt.plot(pendulumMeasurement.transpose() - outputEstimate.transpose(),'--',  label='estimate error')  
plt.legend(loc="upper right")
plt.grid('on')


        
# Run Kalman observer on simulated measurement with intermitant measurements
measurementValuesAvailable = np.full((nSamps), True)
measurementValuesAvailable[200:400] = False
measurementValuesAvailable[600:800] = False

(stateEstimate, outputEstimate) = kalman.propagate(inputVal, pendulumMeasurement, measurementValuesAvailable)

partialMeas = pendulumMeasurement.copy()
partialMeas[:,measurementValuesAvailable==False] = np.nan

plt.figure()  
plt.title('Mesurement not always available')  
plt.plot(partialMeas.transpose(),'.', label='partial measurement') 
plt.plot(outputEstimate.transpose(), label='output estimate')  

plt.plot(pendulumMeasurement.transpose() - outputEstimate.transpose(),'--',   label='estimate error')  
plt.legend(loc="upper right")
plt.grid('on')


# Run Kalman observer on simulated measurement with intermitant and quatified measurements
measurementValuesAvailable = np.full((nSamps), True)
measurementValuesAvailable[200:400] = False
measurementValuesAvailable[600:800] = False

pendulumMeasurementQuantified = np.round(3*pendulumMeasurement) / 3

(stateEstimate, outputEstimate) = kalman.propagate(inputVal, pendulumMeasurementQuantified, measurementValuesAvailable)

partialMeas = pendulumMeasurementQuantified.copy()
partialMeas[:,measurementValuesAvailable==False] = np.nan

plt.figure()  
plt.title('Mesurement quantized and not always available')  
plt.plot(partialMeas.transpose(),'.', label='quantized partial measurement') 
plt.plot(outputEstimate.transpose(), label='output estimate')  

plt.plot(pendulumMeasurement.transpose() - outputEstimate.transpose(),'--',   label='estimate error')  
plt.legend(loc="upper right")
plt.grid('on')
        
        