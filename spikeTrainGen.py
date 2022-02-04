import torch
import numpy as np
import snn_delay_utils
import random
import argparse
import singleNeuron
import pathlib

device = 'gpu'

print(torch.__version__)

# --- Parse Arguments --- #
parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed", default=0, action='store', help="seed", type=int)
parser.add_argument("-nin", "--nin", dest="n_in", default=2000, action='store', help="N_in", type=int)
parser.add_argument("-sdtMax", "--sdtMax", dest="sdtMax", default=50, action='store', help="sdtMax", type=int)
parser.add_argument("-pdtMax", "--pdtMax", dest="pdtMax", default=50, action='store', help="pdtMax", type=int)
parser.add_argument("-patLen", "--patLen", dest="patLen", default=50, action='store', help="pat Len", type=int)
parser.add_argument("-t", "--t", dest="t", default=120000, action='store', help="time", type=int)
parser.add_argument("-rMin", "--rMin", dest="rMin", default=0, action='store', help="R Min", type=int)
parser.add_argument("-rMax", "--rMax", dest="rMax", default=90, action='store', help="R Max", type=int)
parser.add_argument("-fr", "--fr", dest="fr", default=200, action='store', help="frecuencia", type=int)
parser.add_argument("-np", "--np", dest="np", default=1, action='store', help="Cantidad de patrones", type=int)

args = parser.parse_args()

# seed random for same sequence
singleNeuron.seed_torch(seed=args.seed)

seed = args.seed
N_in = args.n_in
Sdt_max = args.sdtMax  # Cada cuanto forzamos un spike (en el tren de spikes)
Pdt_max = args.pdtMax  # Cada cuanto forzamos un spike (en el patron)
pat_len = args.patLen  # longitud del patron
N_pattern = int(N_in / 2)  # Cantidad de neuronas que utiliza el patron
T = args.t
r_min = args.rMin  # Frecuencia minima de spike
r_max = args.rMax  # Frecuencia maxima de spike
fr = args.fr  # Frecuencia con la que aparece el patron
num_pattern = args.np  # Cantidad de patrones a insertar

# Generamos los spikes para training
spike_generator = snn_delay_utils.SpikeTrains(N_in, r_min, r_max, delta_max=Sdt_max)
spike_generator.add_spikes(T)
Sin, s, t = spike_generator.get_spikes(torch.zeros(T, N_in))
pat_times = torch.zeros(T)

# Generamos los spikes para testing
Sin_test = Sin.clone().detach()
pat_times_test = torch.zeros(T)

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
            pat_times[j: j + pat_len] = 1

    # Ponemos el patron en posiciones random
    noisySpikes = random.choice(range(1, 15))
    for j in range(i, T, pat_len):
        if noisySpikes == 0:
            c_index = torch.tensor(range(pat_len))                  # Generamos el indice (0, p_len)
            tmp = Sin_test[j:pat_len + j].clone().detach()          # Copiamos la porcion del tren de spike
            tmp[:, index_inv] = 0                                   # Anulamos las neuronas segun index_A_inv
            Pattern[:, index] = 0                                   # Anulamos las neuronas segun index_A
            Pattern.index_add_(0, c_index, tmp)                     # Copiamos las neuronas
            c_index = torch.tensor(range(0 + j, pat_len + j, 1))    # Generamos el indice dinamico
            Sin_test.index_copy_(0, c_index, Pattern)               # Pegamos el patron
            pat_times_test[j: j + pat_len] = 1
            noisySpikes = random.choice(range(1, 20))               # Elegimos la proxima cantidad a esperar
        else:
            noisySpikes -= 1

# Pre-procesamos PSpikes y NSpikes
dt_ltp = pat_len / 2  # Cantidad de timesteps que miro hacia atras
dt_ltd = pat_len * 3  # Cantidad de timesteps que miro hacia delante
PSpikes = singleNeuron.preSpikes(T, dt_ltp, torch.zeros(T, N_in), Sin)
NSpikes = singleNeuron.nextSpikes(T, dt_ltd, torch.zeros(T, N_in), Sin)

PSpikes_test = singleNeuron.preSpikes(T, dt_ltp, torch.zeros(T, N_in), Sin_test)
NSpikes_test = singleNeuron.nextSpikes(T, dt_ltd, torch.zeros(T, N_in), Sin_test)

pathlib.Path("./spike_trains/" + str(seed)).mkdir(parents=True, exist_ok=True)
# Save Training sequence
print("Save train sequence")
torch.save(Sin, "./spike_trains/" + str(seed) + 'sin.pt')
torch.save(PSpikes, "./spike_trains/" + str(seed) + 'pSpikes.pt')
torch.save(NSpikes, "./spike_trains/" + str(seed) + 'nSpikes.pt')
torch.save(pat_times, "./spike_trains/" + str(seed) + 'pat_times.pt')

# Save Training sequence
print("Save testing sequence")
torch.save(Sin_test, "./spike_trains/" + str(seed) + 'sin_test.pt')
torch.save(PSpikes_test, "./spike_trains/" + str(seed) + 'pSpikes_test.pt')
torch.save(NSpikes_test, "./spike_trains/" + str(seed) + 'nSpikes_test.pt')
torch.save(pat_times_test, "./spike_trains/" + str(seed) + 'pat_times_test.pt')
