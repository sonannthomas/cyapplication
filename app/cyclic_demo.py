import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.io

import toy_example as toy
import cyclic_analysis as cal

gauss = lambda x: np.exp(-2.9*x**2)
sin = np.sin

def create_warped_time(interval, N, peaks):
	start,end = interval
	alpha = 0.05*N/(end-start)
	
	
	pfun = lambda x: np.exp(-alpha*x**2)	
	
	bump_locations = start+np.random.rand(peaks)*(end-start) # Rescale bump size
	bump_amplitudes = -0.5 + 5.5*np.random.rand(peaks) # between 1/2x to 2x faster
	
	tau = np.linspace(start,end,N-1)
	tau = tau[:,np.newaxis]
	bump_locations = bump_locations[np.newaxis,:]
	bump_amplitudes = bump_amplitudes[np.newaxis,:]
	
	dt = 1.+np.sum(pfun(tau-bump_locations)*bump_amplitudes,axis=1)	
	t = np.zeros(N)
	t[1:] = np.cumsum(dt)
	t[1:] = start+(end-start)*t[1:]/t[-1]
	t[0] = start
		
	return t


def create_periodic_stack(wavefun, time, shifts, interval = (-0.5, 0.5)):
	# wave is a callable object defining a waveform on interval
	shifts = shifts[:,np.newaxis]
	time = time[np.newaxis,:]
	
	phase = interval[0]+np.mod(time-shifts-interval[0],interval[1]-interval[0])
		
	return wavefun(phase)  # broadcasing will take care of "outer" sum

def create_cyclic_stack(wavefun, time, epoch, shifts):
	# wave is a callable object defining a waveform on [0,1] 
	shifts = shifts[:,np.newaxis,np.newaxis]
	time = time[np.newaxis,:,np.newaxis]	
	epoch = epoch[np.newaxis,np.newaxis,:]
	w = wavefun(time-epoch-shifts)  # broadcasing will take care of "outer" sum
	return w.sum(axis=2)


#default = toy.descriptor()
#lines_p = toy.create_toy_periodic(default.rois, default.time, default.shift, default.freq)
#lines_c = toy.create_toy_cyclic(default.rois, default.time, default.shift, default.freq)
#lines_g = toy.create_toy_gaussian(default.rois, default.time, default.shift, default.peaks)
#sort_indices = np.argsort(default.shift)[::-1] # Reverse sort


channels = 5
T_max = 50
N = 25
#epoch = np.random.rand(channels)*T_max

#T = np.linspace(0,T_max,1000)
T = create_warped_time((0,T_max),N,10)
amps = 1.+np.random.randn(channels)
amps = amps[:,np.newaxis] # Amplitudes

shifts = 3*np.random.rand(channels)

#lines = amps*create_cyclic_stack(gauss, T, epoch, shifts)+np.random.randn(channels,1000)*0.03
lines = amps*create_periodic_stack(gauss, T, shifts,(-3,3))+np.random.randn(channels,N)*0.02

k = 0
for line in lines:
	color = np.array([0.4,1,1])*shifts[k]/3 + np.array([0.4,0,1])*(1-shifts[k]/3)
	plt.plot(line,linewidth=2.0,color = color)
	k+=1
	
#plt.axis('off')
#plt.show()

chlist = ['star' + str(k)  for k in range(channels)]
header = ','.join(chlist)
np.savetxt('out.csv',lines.T,header= header, delimiter=',')

gg = np.genfromtxt('out.csv',delimiter=',',names=True)
print(gg[0])

print(gg.dtype.names)
	


