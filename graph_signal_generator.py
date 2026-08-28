import matplotlib.pyplot as plt
import numpy as np
from graph_topology_generator import GraphTopology


class GraphSignalGenerator:

    def __init__(self, topology: GraphTopology):

        self.topology = topology
        self.N = topology.N

        if self.topology.U is None:
            self.topology.compute_fourier_basis()

    def generate_bandlimited_signal(self, cutoff_idx=5, seed=None):
        """Genera un segnale esattamente a banda limitata usando i primi 'cutoff_idx' autovettori."""
        
        if seed is not None:
            np.random.seed(seed)

        c = np.random.randn(cutoff_idx)
        x_smooth = self.topology.U[:, :cutoff_idx] @ c

        return x_smooth

    def add_noise(self, x_clean, snr_db=10.0, seed=None):
        """Aggiunge rumore bianco gaussiano (AWGN) per ottenere uno specifico SNR in dB:

        SNR_dB = 10 * log10(P_signal / P_noise)
        """
        if seed is not None:
            np.random.seed(seed)

        p_signal = np.mean(x_clean**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        p_noise = p_signal / snr_linear

        noise = np.random.normal(0.0, np.sqrt(p_noise), size=self.N)
        x_noisy = x_clean + noise
        return x_noisy, noise