import matplotlib.pyplot as plt
import numpy as np

from graph_filter_ls import GraphFilterLS
from graph_signal_generator import GraphSignalGenerator
from graph_topology_generator import GraphTopology


def compute_snr(signal_clean, signal_noisy_or_filtered):
    """Calcola l'SNR in dB."""
    noise = signal_noisy_or_filtered - signal_clean
    p_signal = np.mean(signal_clean**2)
    p_noise = np.mean(noise**2)
    if p_noise == 0:
        return np.inf
    return 10.0 * np.log10(p_signal / p_noise)


def compute_mse(signal_clean, signal_noisy_or_filtered):
    """Calcola l'Errore Quadratico Medio (MSE)."""
    return np.mean((signal_clean - signal_noisy_or_filtered) ** 2)


def run_grid_search():
    # ---------------------------------------------------------
    # 1. PARAMETRI DELLA GRID SEARCH
    # ---------------------------------------------------------
    snr_in_list = np.array([0, 5, 8, 12, 16, 20])   # Rumore in ingresso (dB)
    k_orders = np.array([2, 4, 6, 8, 12])            # Ordini del filtro polinomiale
    cutoff_idx = 5

    # Topologia fissa
    g = GraphTopology(num_nodes=50, k_neighbors=4, seed=42)
    sig_gen = GraphSignalGenerator(g)
    lambda_cutoff_oracle = g.eigenvalues[cutoff_idx - 1]

    def ideal_lowpass_oracle(lam):
        return 1.0 if lam <= lambda_cutoff_oracle else 0.0

    # Matrici dei risultati (Dimensioni: K x SNR_in)
    # Matrici SNR
    snr_res_oracle = np.zeros((len(k_orders), len(snr_in_list)))
    snr_res_energy = np.zeros((len(k_orders), len(snr_in_list)))

    # Matrici MSE
    mse_res_oracle = np.zeros((len(k_orders), len(snr_in_list)))
    mse_res_energy = np.zeros((len(k_orders), len(snr_in_list)))

    print(f"Avvio Grid Search: {len(k_orders)} ordini K x {len(snr_in_list)} valori SNR...")

    # ---------------------------------------------------------
    # 2. ESECUZIONE DELLA SIMULAZIONE
    # ---------------------------------------------------------
    for i, K in enumerate(k_orders):
        for j, snr_target in enumerate(snr_in_list):
            
            # Generazione singola di segnale pulito e rumoroso per il punto (K, SNR_in)
            seed_point = i * 100 + j
            x_clean = sig_gen.generate_bandlimited_signal(cutoff_idx=cutoff_idx, seed=seed_point)
            x_noisy, _ = sig_gen.add_noise(x_clean, snr_db=snr_target, seed=seed_point + 1)

            # --- 1) LS Spettrale Oracolo ---
            f_oracle = GraphFilterLS(g, order=K, reg_param=1e-5)
            f_oracle.fit_spectral(ideal_lowpass_oracle)
            x_out_oracle = f_oracle.filter(x_noisy)
            snr_res_oracle[i, j] = compute_snr(x_clean, x_out_oracle)
            mse_res_oracle[i, j] = compute_mse(x_clean, x_out_oracle)

            # --- 2) LS Spettrale Energia (75%) ---
            gft_energy = (g.U.T @ x_noisy) ** 2
            cum_energy = np.cumsum(gft_energy) / np.sum(gft_energy)
            cutoff_idx_energy = np.searchsorted(cum_energy, 0.75)
            lambda_cutoff_energy = g.eigenvalues[cutoff_idx_energy]

            def ideal_lowpass_energy(lam, l_c=lambda_cutoff_energy):
                return 1.0 if lam <= l_c else 0.0

            f_energy = GraphFilterLS(g, order=K, reg_param=1e-5)
            f_energy.fit_spectral(ideal_lowpass_energy)
            x_out_energy = f_energy.filter(x_noisy)
            snr_res_energy[i, j] = compute_snr(x_clean, x_out_energy)
            mse_res_energy[i, j] = compute_mse(x_clean, x_out_energy)

        print(f" -> Completato K = {K:2d}")

    # ---------------------------------------------------------
    # 3. VISUALIZZAZIONE DEI RISULTATI (2 Righe x 2 Colonne)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- RIGA 1: SNR OUT vs SNR IN ---
    # [1.1] Oracolo - SNR
    ax = axes[0, 0]
    for i, K in enumerate(k_orders):
        ax.plot(snr_in_list, snr_res_oracle[i, :], marker='o', label=f'K = {K}')
    ax.plot(snr_in_list, snr_in_list, 'k--', alpha=0.5, label='No Filtering')
    ax.set_title("1) Oracolo - $\\text{SNR}_{\\text{out}}$ vs $\\text{SNR}_{\\text{in}}$", fontsize=11)
    ax.set_xlabel(r"$\text{SNR}_{\text{in}}$ (dB)")
    ax.set_ylabel(r"$\text{SNR}_{\text{out}}$ (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

    # [1.2] Energia 75% - SNR
    ax = axes[0, 1]
    for i, K in enumerate(k_orders):
        ax.plot(snr_in_list, snr_res_energy[i, :], marker='^', linestyle=':', label=f'K = {K}')
    ax.plot(snr_in_list, snr_in_list, 'k--', alpha=0.5, label='No Filtering')
    ax.set_title("2) Energia (75%) - $\\text{SNR}_{\\text{out}}$ vs $\\text{SNR}_{\\text{in}}$", fontsize=11)
    ax.set_xlabel(r"$\text{SNR}_{\text{in}}$ (dB)")
    ax.set_ylabel(r"$\text{SNR}_{\text{out}}$ (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- RIGA 2: MSE OUT (Scala Semilogaritmica) vs SNR IN ---
    # [2.1] Oracolo - MSE
    ax = axes[1, 0]
    for i, K in enumerate(k_orders):
        ax.semilogy(snr_in_list, mse_res_oracle[i, :], marker='o', label=f'K = {K}')
    ax.set_title("1) Oracolo - MSE di Ricostruzione", fontsize=11)
    ax.set_xlabel(r"$\text{SNR}_{\text{in}}$ (dB)")
    ax.set_ylabel("MSE (Scala Logaritmica)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

    # [2.2] Energia 75% - MSE
    ax = axes[1, 1]
    for i, K in enumerate(k_orders):
        ax.semilogy(snr_in_list, mse_res_energy[i, :], marker='^', linestyle=':', label=f'K = {K}')
    ax.set_title("2) Energia (75%) - MSE di Ricostruzione", fontsize=11)
    ax.set_xlabel(r"$\text{SNR}_{\text{in}}$ (dB)")
    ax.set_ylabel("MSE (Scala Logaritmica)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_grid_search()