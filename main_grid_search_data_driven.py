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


def run_grid_search_datadriven():
    # ---------------------------------------------------------
    # 1. PARAMETRI DELLA GRID SEARCH
    # ---------------------------------------------------------
    snr_in_list = np.array([0, 5, 8, 12, 16, 20])   # Rumore in ingresso (dB)
    k_orders = np.array([2, 4, 6, 8, 12])            # Ordini del filtro polinomiale
    cutoff_idx = 5
    reg_mu = 1e-3                                    # Parametro regolarizzazione Tikhonov

    # Topologia fissa del grafo
    g = GraphTopology(num_nodes=50, k_neighbors=4, seed=42)
    sig_gen = GraphSignalGenerator(g)

    # Matrici dei risultati (solo Test Set)
    snr_res_test = np.zeros((len(k_orders), len(snr_in_list)))
    mse_res_test = np.zeros((len(k_orders), len(snr_in_list)))

    print("=" * 70)
    print(f"Avvio Grid Search Data-Driven: {len(k_orders)} ordini K x {len(snr_in_list)} valori SNR...")
    print("=" * 70)

    # ---------------------------------------------------------
    # 2. ESECUZIONE DELLA SIMULAZIONE
    # ---------------------------------------------------------
    for i, K in enumerate(k_orders):
        for j, snr_target in enumerate(snr_in_list):
            
            # --- 2.1 Set di Training (Calibrazione dei Pesi) ---
            seed_train = i * 1000 + j * 10
            x_clean_train = sig_gen.generate_bandlimited_signal(cutoff_idx=cutoff_idx, seed=seed_train)
            x_noisy_train, _ = sig_gen.add_noise(x_clean_train, snr_db=snr_target, seed=seed_train + 1)

            # --- 2.2 Set di Test Indipendente (Nuova Istanza) ---
            seed_test = i * 1000 + j * 10 + 500
            x_clean_test = sig_gen.generate_bandlimited_signal(cutoff_idx=cutoff_idx, seed=seed_test)
            x_noisy_test, _ = sig_gen.add_noise(x_clean_test, snr_db=snr_target, seed=seed_test + 1)

            # --- 2.3 Addestramento del Filtro ---
            f_data = GraphFilterLS(g, order=K, reg_param=reg_mu)
            f_data.fit_data_driven(x_noisy_train, x_clean_train)

            # --- 2.4 Valutazione su Test Set ---
            x_out_test = f_data.filter(x_noisy_test)
            snr_res_test[i, j] = compute_snr(x_clean_test, x_out_test)
            mse_res_test[i, j] = compute_mse(x_clean_test, x_out_test)

        print(f" -> Completato K = {K:2d}")

    # ---------------------------------------------------------
    # 3. VISUALIZZAZIONE DEI RISULTATI (1 Riga x 2 Colonne)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # [Pannello 1]: SNR OUT vs SNR IN (Test Set)
    ax = axes[0]
    for i, K in enumerate(k_orders):
        ax.plot(snr_in_list, snr_res_test[i, :], marker='o', label=f'K = {K}')
    ax.plot(snr_in_list, snr_in_list, 'k--', alpha=0.5, label='No Filtering')
    ax.set_title(r"Data-Driven - $\text{SNR}_{\text{out}}$ vs $\text{SNR}_{\text{in}}$ (Test Set)", fontsize=11)
    ax.set_xlabel(r"$\text{SNR}_{\text{in}}$ (dB)")
    ax.set_ylabel(r"$\text{SNR}_{\text{out}}$ (dB)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    # [Pannello 2]: MSE OUT (Scala Semilogaritmica) vs SNR IN (Test Set)
    ax = axes[1]
    for i, K in enumerate(k_orders):
        ax.semilogy(snr_in_list, mse_res_test[i, :], marker='o', label=f'K = {K}')
    ax.set_title("Data-Driven - MSE di Ricostruzione (Test Set)", fontsize=11)
    ax.set_xlabel(r"$\text{SNR}_{\text{in}}$ (dB)")
    ax.set_ylabel("MSE (Scala Logaritmica)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_grid_search_datadriven()