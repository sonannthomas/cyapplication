# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 22:01:43 2016

@author: elady
"""

import numpy as np
from numpy.random import rand, randn, randint
from numpy import linspace, sin, pi, argsort, zeros
from numpy.matlib import repmat
import matplotlib.pyplot as plt

import scipy
from scipy.interpolate import splev, splrep


class descriptor:
	rois = 6
	freq = 5
	time = 1000
	noise = 0
	shift = 12*rand(rois)
	peaks = 4
	

def add_noise(lines,noise,time):
	return lines + noise * repmat(np.std(lines, axis=1), 1, time) * randn(lines.shape[0],lines.shape[1]);


def create_toy_periodic(rois,time,shifts,freq):
	shifts = shifts/time # Normalize shifts
	tt = shifts[:,np.newaxis]+ linspace(0,1,time)
	lines = sin(2*np.pi * freq * tt)
	return lines
	
					
def create_toy_cyclic(rois,time,shifts,freq):	
	T = time*100	
	shifts = np.round(100*shifts)
	while len(set(shifts)) != rois:
		shifts = np.round(100*shifts*rand(rois))
	
	
	L=np.ceil(T+max(shifts)).astype(np.int32)
	tau = zeros(L)
	stepSize = 2*pi*freq/T
	
	tau[0] = stepSize
	for i in range(1,L):
		if (sin(tau[i-1])+1<0.01) and (rand()< 0.98):
			tau[i] = tau[i-1]
		elif sin(tau[i-1]+4*stepSize) < sin(tau[i-1]):
			tau[i] = tau[i-1] + 2*stepSize
		else:
			tau[i] = tau[i-1] + 6*stepSize
			
			
	lines = zeros((rois,time))	

	for i in range(rois):
		start = shifts[i]
		lines[i] = sin(tau[start:start+T:100])
	return lines


def gaussian(x,mu,sigma):
	return (1/np.sqrt(2*pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))


def create_toy_gaussian(rois,time,shifts,peaks):	
	t = 1.+ np.arange(time)
	
	# Create a sum of gaussians
	line = 30. * gaussian(t,time/2,1)+500
	for i in range(peaks-1):
		sigma = randint(2,peaks*2-1)/(2*peaks-1)
		line += 100*rand()*gaussian(t,sigma*time,randint(1,4))

	f = splrep(t,line)

	# Multiplex - each row gets a different shift	
	lines = zeros((rois,time))	
	for i in range(rois):
		lines[i] = splev(t+shifts[i],f)

	return lines


