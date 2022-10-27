# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:31:54 2016

@author: elady
"""

import numpy as np
from numpy import mod, outer, mean, argsort
from numpy import pi, linspace, newaxis, roll, zeros, angle
from numpy.linalg import norm, eig


def match_ends(Z):
	n = Z.shape[1] # No of columns
	return Z-outer(Z[:,n-1]-Z[:,0],linspace(0,1,n))
	
def mean_center(Z):
	return Z - mean(Z,axis=1)[:,newaxis]

def total_variation(Z,axis=0):
	dZ = Z-roll(Z,1,axis=axis)
	return norm(dZ,axis=axis).squeeze()

# Normalize vector(s) Z so that the quadratic sum equals to 1.
def quad_norm(Z):
	norms = norm(Z,axis=1)	
	normed_z = Z/norms[:,newaxis]
	return mean_center(normed_z)

# Normalize vector(s) Z so that the quadratic variation is 1. See
# Frobenius norm
def tv_norm(Z):
	norms = total_variation(Z,axis=1)
	normed_z = Z/norms[:,newaxis]
	return mean_center(normed_z)

def remove_bias(Z):
	return mean_center(Z)

def remove_linear(Z):
	return match_ends(mean_center(Z))


def cyc_diff(x):
	return np.diff(np.concatenate(([x[-1]],x)))

def create_lead_matrix(data):
	N,time_steps = data.shape
	lead_matrix = zeros((N,N))
	
	# Create index list of upper_trinagle (lower part is anti-symmetric)
	# This is good for small values of N.
	# For larger values, just do a double "for" loop
	upper_triangle = [(i,j) for i in range(N) for j in range(i+1,N)]
	
	for (i,j) in upper_triangle: 
		x,y = data[i],data[j]
		d = x.dot(cyc_diff(y)) - y.dot(cyc_diff(x))
		lead_matrix[i,j] = d
		lead_matrix[j,i] = -d

	return time_steps*lead_matrix

# LM: lead matrix (square matrix to be sorted)
# p: eigenvector index to use (0...)
def sort_lead_matrix(LM,p=1):
# The first input should be the matrix to be sorted, the second is the
# phase or eigenvector to use (default 1).
	evals,phases = eig(LM)
	phases = phases[:,2*p-2]
	sorted_ang = np.sort(mod(angle(phases),2*pi))
	dang = np.diff(np.hstack((sorted_ang,sorted_ang[0]+2*pi)))
	shift = np.argmax(np.abs(dang))
	shift= (shift+1) % phases.size
		
	shift = sorted_ang[shift]
	perm = argsort(mod(mod(angle(phases),2*pi)-shift,2*pi))
	print(perm)
	sorted_LM = LM[perm].T[perm].T
	
	return LM, phases, perm, sorted_LM, evals

def cyclic_analysis(data,p, normalize, trend_removal):	
	normed_data = trend_removal(normalize(data))
	lead_matrix = create_lead_matrix(normed_data)
	return (sort_lead_matrix(lead_matrix, p),normed_data)
