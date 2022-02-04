import pathlib

import torch
import numpy as np
import snn_delay_utils
import argparse
import singleNeuron
device = 'gpu'

print(torch.__version__)

# --- Parse Arguments --- #
parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", dest="seed", default=0, action='store', help="seed", type=int)
parser.add_argument("-nin", "--nin", dest="n_in", default=2000, action='store', help="N_in", type=int)
parser.add_argument("-nout", "--nout", dest="n_out", default=1, action='store', help="N_out", type=int)
parser.add_argument("-patLen", "--patLen", dest="patLen", default=50, action='store', help="pat Len", type=int)
parser.add_argument("-t", "--t", dest="t", default=60000, action='store', help="time", type=int)
parser.add_argument("-fr", "--fr", dest="fr", default=200, action='store', help="frecuencia", type=int)
parser.add_argument("-th", "--th", dest="th", default=625, action='store', help="threshold", type=float)
parser.add_argument("-ap", "--ap", dest="aPlus", default="0.009125", action='store', help="a_plus", type=str)
parser.add_argument("-am", "--am", dest="aMinus", default="0.0125625", action='store', help="a_minus", type=str)

args = parser.parse_args()

# seed random for same sequence
singleNeuron.seed_torch(seed=args.seed)
seed = args.seed

N_in = args.n_in
N_out = args.n_out
T = args.t
th = args.th  # Threshold de la neurona
a_plus = float(args.aPlus)
a_minus = float(args.aMinus)
fr = args.fr  # Frecuencia con la que aparece el patron
pat_len = args.patLen  # longitud del patron

# Realizamos el entrenamiento STDP
pop1 = singleNeuron.STDPLIFDensePopulation(in_channels=N_in, out_channels=N_out,
                                           weight=0.475, alpha=float(np.exp(-1e-3 / 10e-3)),
                                           beta=float(np.exp(-1e-3 / 2e-5)), delay=0,
                                           th=th,
                                           a_plus=a_plus, a_minus=a_minus,
                                           w_max=1.)

# Pre-procesamos PSpikes y NSpikes
dt_ltp = pat_len / 2  # Cantidad de timesteps que miro hacia atras
dt_ltd = pat_len * 3  # Cantidad de timesteps que miro hacia delante
PSpikes = np.load("./spike_trains/" + str(seed) + 'pSpikes.pt')
NSpikes = np.load("./spike_trains/" + str(seed) + 'nSpikes.pt')

Sin = np.load("./spike_trains/" + str(seed) + 'sin.pt')

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
    if n % 5000 == 0:
        c_l = sum(w_end * (1 - w_end)) / len(w_end)
        print(str(c_l))


# Realizamos el testing
# Pre-procesamos PSpikes y NSpikes
dt_ltp = pat_len / 2  # Cantidad de timesteps que miro hacia atras
dt_ltd = pat_len * 3  # Cantidad de timesteps que miro hacia delante
PSpikes_test = np.load("./spike_trains/" + str(seed) + 'pSpikes_test.pt')
NSpikes_test = np.load("./spike_trains/" + str(seed) + 'nSpikes_test.pt')

Sin_test = np.load("./spike_trains/" + str(seed) + 'sin_test.pt')
pat_times_test = np.load("./spike_trains/" + str(seed) + 'pat_times_test.pt')

# Realizamos el entrenamiento STDP
Uprobe = np.empty([T, N_out])
Iprobe = np.empty([T, N_out])
Sprobe = np.empty([T, N_out])
for n in range(T):
    state = pop1.forward(Sin[n].unsqueeze(0), PSpikes_test[n], NSpikes_test[n - 1])
    Uprobe[n] = state.U.data.numpy()
    Iprobe[n] = state.I.data.numpy()
    Sprobe[n] = state.S.data.numpy()


# Corremos metricas de testing
accuracy, precision, recall, f1, fake_alarms, missed_alarms = singleNeuron.get_metrics_long_pat(T, pat_times_test, Sprobe, pat_len)
metrics = np.array([accuracy, precision, recall, f1, fake_alarms, missed_alarms], dtype=float)
pathlib.Path("./spike_trains/" + str(seed)).mkdir(parents=True, exist_ok=True)
np.save("./spike_trains/" + str(seed) + 'metrics_test.npy', metrics)