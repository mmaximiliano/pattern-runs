import torch
import numpy as np
import snn_delay_utils
import random
import argparse
import singleNeuron
device = 'cpu'

print(torch.__version__)

# --- Parse Arguments --- #
parser = argparse.ArgumentParser()
parser.add_argument("-nin", "--nin", dest="n_in", default=2000, action='store', help="N_in", type=int)
parser.add_argument("-nout", "--nout", dest="n_out", default=1, action='store', help="N_out", type=int)
parser.add_argument("-sdtMax", "--sdtMax", dest="sdtMax", default=50, action='store', help="sdtMax", type=int)
parser.add_argument("-pdtMax", "--pdtMax", dest="pdtMax", default=50, action='store', help="pdtMax", type=int)
parser.add_argument("-patLen", "--patLen", dest="patLen", default=50, action='store', help="pat Len", type=int)
parser.add_argument("-t", "--t", dest="t", default=60000, action='store', help="time", type=int)
parser.add_argument("-rMin", "--rMin", dest="rMin", default=0, action='store', help="R Min", type=int)
parser.add_argument("-rMax", "--rMax", dest="rMax", default=90, action='store', help="R Max", type=int)
parser.add_argument("-fr", "--fr", dest="fr", default=200, action='store', help="frecuencia", type=int)
parser.add_argument("-th", "--th", dest="th", default=625, action='store', help="threshold", type=int)
parser.add_argument("-np", "--np", dest="np", default=1, action='store', help="Cantidad de patrones", type=int)

args = parser.parse_args()

N_in = args.n_in
N_out = args.n_out
Sdt_max = args.sdtMax  # Cada cuanto forzamos un spike (en el tren de spikes)
Pdt_max = args.pdtMax  # Cada cuanto forzamos un spike (en el patron)
pat_len = args.patLen  # longitud del patron
N_pattern = int(N_in / 2)  # Cantidad de neuronas que utiliza el patron
T = args.t
r_min = args.rMin  # Frecuencia minima de spike
r_max = args.rMax  # Frecuencia maxima de spike
fr = args.fr  # Frecuencia con la que aparece el patron
th = args.th  # Threshold de la neurona
num_pattern = args.np  # Cantidad de patrones a insertar

# Generamos los spikes
spike_generator = snn_delay_utils.SpikeTrains(N_in, r_min, r_max, delta_max=Sdt_max)
spike_generator.add_spikes(T)
Sin, s, t = spike_generator.get_spikes(torch.zeros(T, N_in))

# Evaluate the mean firing rate
rate = np.count_nonzero(spike_generator.spikes) * 1000.0 / T / N_in
print("Spike train rate: " + str(rate))

# Pegamos el patron a lo largo del tren de spikes
for i in range(num_pattern):
    # Generamos un patron con el mismo rate
    index = random.sample(range(N_in), N_pattern)
    index_inv = singleNeuron.find_missing(index, N_in)  # Neuronas involucradas en el Patron

    Pattern_base = Sin[150 + (i * 1000):200 + (i * 1000), :].clone().detach()
    Pattern_base[:, index] = 0
    Pattern = Pattern_base.clone().detach()
    rate = np.count_nonzero(Pattern[:, index_inv]) * 1000.0 / pat_len / N_pattern
    print('Pattern ' + str(i) + ': Resulting mean rate: %d Hz' % rate)

    for j in range(fr * i, T, fr * num_pattern):
        if ((j + pat_len) < T):
            c_index = torch.tensor(range(pat_len))  # Generamos el indice (0, p_len)
            tmp = Sin[j:pat_len + j].clone().detach()  # Copiamos la porcion del tren de spike
            tmp[:, index_inv] = 0  # Anulamos las neuronas segun index_A_inv
            Pattern[:, index] = 0  # Anulamos las neuronas segun index_A
            Pattern.index_add_(0, c_index, tmp)  # Copiamos las neuronas
            c_index = torch.tensor(range(0 + j, pat_len + j, 1))  # Generamos el indice dinamico
            Sin.index_copy_(0, c_index, Pattern)  # Pegamos el patron

# Realizamos el entrenamiento STDP
pop1 = singleNeuron.STDPLIFDensePopulation(in_channels=N_in, out_channels=N_out,
                                           weight=0.45, alpha=float(np.exp(-1e-3 / 10e-3)),
                                           beta=float(np.exp(-1e-3 / 2e-5)), delay=0,
                                           th=th,
                                           a_plus=0.05125, a_minus=0.0865625,
                                           w_max=1.)

# Pre-procesamos PSpikes y NSpikes
dt_ltp = pat_len / 2  # Cantidad de timesteps que miro hacia atras
dt_ltd = pat_len * 3  # Cantidad de timesteps que miro hacia delante
PSpikes = singleNeuron.preSpikes(T, dt_ltp, torch.zeros(T, N_in), Sin)
NSpikes = singleNeuron.nextSpikes(T, dt_ltd, torch.zeros(T, N_in), Sin)

# Realizamos el entrenamiento STDP
Uprobe = np.empty([T, N_out])
Iprobe = np.empty([T, N_out])
Sprobe = np.empty([T, N_out])
for n in range(T):
    state = pop1.forward(Sin[n].unsqueeze(0), PSpikes[n], NSpikes[n - 1])
    Uprobe[n] = state.U.data.numpy()
    Iprobe[n] = state.I.data.numpy()
    Sprobe[n] = state.S.data.numpy()
    w_end = pop1.fc_layer.weight.data[0].detach().numpy()
    if n % 1000:
        c_l = sum(w_end * (1 - w_end)) / len(w_end)
        print(str(c_l))

# Realizamos el testing
#Uprobe = np.empty([T, N_out])
#Iprobe = np.empty([T, N_out])
#Sprobe = np.empty([T, N_out])
#for n in range(T):
#    state = pop1.forward_no_learning(Sin[n].unsqueeze(0))
#    Uprobe[n] = state.U.data.numpy()
#    Iprobe[n] = state.I.data.numpy()
#    Sprobe[n] = state.S.data.numpy()
#    w_end = pop1.fc_layer.weight.data[0].detach().numpy()
