import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms, utils
import torch
from torch.utils.data.dataloader import DataLoader
#import tqdm
import copy
from collections import namedtuple


def spikes_to_evlist(spikes):
    t = np.tile(np.arange(spikes.shape[0]), [spikes.shape[1],1])
    n = np.tile(np.arange(spikes.shape[1]), [spikes.shape[0],1]).T  
    return t[spikes.astype('bool').T], n[spikes.astype('bool').T]

def plotLIF(U, S, Vplot = 'all', staggering= 1, ax1=None, ax2=None, pat_times=None, th=None, **kwargs):
    '''
    This function plots the output of the function LIF.
    
    Inputs:
    *S*: an TxNnp.array, where T are time steps and N are the number of neurons
    *S*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF
    *Vplot*: A list indicating which neurons' membrane potentials should be 
    plotted. If scalar, the list range(Vplot) are plotted. Default: 'all'
    *staggering*: the amount by which each V trace should be shifted. None
    
    Outputs the figure returned by figure().    
    '''
    V = U
    spikes = S
    #Plot
    t, n = spikes_to_evlist(spikes)
    #f = plt.figure()
    if V is not None and ax1 is None:
        ax1 = plt.subplot(211)
    elif V is None:
        ax1 = plt.axes()
        ax2 = None
    ax1.plot(t, n, 'k|', **kwargs)
    ax1.set_ylim([-1, spikes.shape[1] + 1])
    ax1.set_xlim([0, spikes.shape[0]])

    if V is not None:
        if Vplot == 'all':
            Vplot = range(V.shape[1])
        elif not hasattr(Vplot, '__iter__'):
            Vplot = range(np.minimum(Vplot, V.shape[1]))    
        
        if ax2 is None:
            ax2 = plt.subplot(212)
    
        if V.shape[1]>1:
            for i, idx in enumerate(Vplot):
                ax2.plot(V[:,idx]+i*staggering,'-',  **kwargs)
        else:
            ax2.plot(V[:,0], '-', **kwargs)
            
        if staggering!=0:
            plt.yticks([])
        plt.xlabel('time [ms]')
        plt.ylabel('u [au]')

    if pat_times is not None:
      for i in range(0, spikes.shape[0]):
        if pat_times[i]:
            plt.axvspan(i, i+1, facecolor='lightgray')

    if th is not None:
      plt.hlines(th, 0, spikes.shape[0], 'red', 'dashed')

    ax1.set_ylabel('Neuron ')

    plt.xlim([0, spikes.shape[0]])
    plt.ion()
    plt.show()
    return ax1,ax2