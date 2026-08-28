import matplotlib.pyplot as plt
import numpy as np
from graph_filter_ls import GraphFilterLS
from graph_signal_generator import GraphSignalGenerator
from graph_topology_generator import GraphTopology


def compute_snr(signal_clean, signal_noisy_or_filtered):
    """Calcola l'SNR in dB tra segnale pulito e segnale stimato/rumoroso."""
    noise = signal_noisy_or_filtered - signal_clean
    p_signal = np.mean(signal_clean**2)
    p_noise = np.mean(noise**2)
    if p_noise == 0:
        return np.inf
    return 10.0 * np.log10(p_signal / p_noise)


# =========================================================================
# ESEMPIO DI UTILIZZO DEL FILTRO DATA-DRIVEN SUI NODI (LEAST SQUARES)
# =========================================================================


if __name__ == '__main__':
    
    # =========================================================================
    # CONFIGURAZIONE
    # =========================================================================

    K_order = 6          # Ordine polinomiale
    cutoff_idx = 5       # Banda del segnale (numero di modi spettrali)
    snr_noise_db = 8.0   # Livello di rumore aggiunto
    reg_param_mu = 1e-3  # Parametro di regolarizzazione Tikhonov mu

    g = GraphTopology(num_nodes=50, k_neighbors=4, seed=42)
    sig_gen = GraphSignalGenerator(g)

    # =========================================================================
    # 1. GENERAZIONE DATI: TRAINING E TEST
    # =========================================================================
    # Set di Calibrazione / Training
    x_clean_train = sig_gen.generate_bandlimited_signal(cutoff_idx=cutoff_idx, seed=10)
    x_noisy_train, _ = sig_gen.add_noise(x_clean_train, snr_db=snr_noise_db, seed=20)
    snr_in_train = compute_snr(x_clean_train, x_noisy_train)

    # Set di Test Indipendente (stessa fisica di banda, diverso seed/rumore)
    x_clean_test = sig_gen.generate_bandlimited_signal(cutoff_idx=cutoff_idx, seed=88)
    x_noisy_test, _ = sig_gen.add_noise(x_clean_test, snr_db=snr_noise_db, seed=99)
    snr_in_test = compute_snr(x_clean_test, x_noisy_test)

    # =========================================================================
    # 2. ADDESTRAMENTO DEL FILTRO DATA-DRIVEN (LEAST SQUARES SUI NODI)
    # =========================================================================
    filter_data = GraphFilterLS(g, order=K_order, reg_param=reg_param_mu)
    
    # Risolve: min_h ||X_train * h - x_clean_train||^2 + mu ||h||^2
    h_opt = filter_data.fit_data_driven(x_in=x_noisy_train, y_target=x_clean_train)
    filter_data.plot_frequency_response(target_filter_fn=None, title="Filtro Data-Driven")

    # =========================================================================
    # 3. FILTRAGGIO E VALUTAZIONE PRESTAZIONI
    # =========================================================================
    # Valutazione su Training (In-Sample)
    x_filtered_train = filter_data.filter(x_noisy_train)
    mse_train = np.mean((x_filtered_train - x_clean_train) ** 2)
    snr_out_train = compute_snr(x_clean_train, x_filtered_train)

    # Valutazione su Test
    x_filtered_test = filter_data.filter(x_noisy_test)
    mse_test = np.mean((x_filtered_test - x_clean_test) ** 2)
    snr_out_test = compute_snr(x_clean_test, x_filtered_test)

    # =========================================================================
    # 4. REPORT A TERMINALE
    # =========================================================================
    print("=" * 65)
    print("ANALISI FILTRO DATA-DRIVEN SU GRAFO (LEAST SQUARES / KRYLOV)")
    print("=" * 65)
    print(f"Ordine del filtro (K-hop): {K_order}")
    print(f"Parametro regolarizzazione (mu): {reg_param_mu}")
    print(f"Coefficienti stimati h*: {np.round(h_opt, 4)}")
    print("-" * 65)
    print("PRESTAZIONI SUL SET DI TRAINING (In-Sample):")
    print(f"  Ingresso Rumoroso: SNR = {snr_in_train:5.2f} dB | MSE = {np.mean((x_noisy_train - x_clean_train)**2):.4f}")
    print(f"  Uscita Filtrata:   SNR = {snr_out_train:5.2f} dB | MSE = {mse_train:.4f} (Guadagno: +{snr_out_train - snr_in_train:4.2f} dB)")
    print("-" * 65)
    print("PRESTAZIONI SUL SET DI TEST (Out-of-Sample / Generalizzazione):")
    print(f"  Ingresso Rumoroso: SNR = {snr_in_test:5.2f} dB | MSE = {np.mean((x_noisy_test - x_clean_test)**2):.4f}")
    print(f"  Uscita Filtrata:   SNR = {snr_out_test:5.2f} dB | MSE = {mse_test:.4f} (Guadagno: +{snr_out_test - snr_in_test:4.2f} dB)")
    print("=" * 65)

    # =========================================================================
    # 5. FIGURA 1 - ANALISI QUANTITATIVA E SPETTRALE (1x2)
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5))
    nodes = np.arange(g.N)

    # [1.1] Segnali sui Nodi (Test Set)
    ax = axes1[0]
    ax.plot(nodes, x_clean_test, 'k-', linewidth=2, label='Ground Truth (Clean)')
    ax.plot(nodes, x_noisy_test, 'r.', alpha=0.5, label=f'Noisy Input ({snr_in_test:.1f} dB)')
    ax.plot(nodes, x_filtered_test, 'g.-', linewidth=1.5, label=f'Data-Driven Output ({snr_out_test:.1f} dB)')
    ax.set_title("Ricostruzione Segnale sui Vertici (Test Set)", fontsize=11)
    ax.set_xlabel("Indice del Nodo")
    ax.set_ylabel("Ampiezza")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    # [1.2] Spettro GFT (Test Set)
    ax = axes1[1]
    gft_clean = np.abs(g.U.T @ x_clean_test)
    gft_noisy = np.abs(g.U.T @ x_noisy_test)
    gft_filt = np.abs(g.U.T @ x_filtered_test)

    ax.stem(nodes, gft_noisy, linefmt='r:', markerfmt='rx', basefmt='k-', label='Noisy')
    ax.stem(nodes, gft_filt, linefmt='g-', markerfmt='go', basefmt='k-', label='Data-Driven')
    ax.stem(nodes, gft_clean, linefmt='k--', markerfmt='k.', basefmt='k-', label='Clean')
    ax.set_title(r"Spettro GFT: $|\hat{x}_i|$ (Attenuazione Alte Frequenze)", fontsize=11)
    ax.set_xlabel("Indice di frequenza ($i$)")
    ax.set_ylabel("Ampiezza Spettrale")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    fig1.tight_layout()

    # =========================================================================
    # 6. FIGURA 2 - CONFRONTO TOPOLOGICO SUL GRAFO (1x3)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    A = g.A_binary.toarray()

    # Normalizzazione comune della scala colore per un confronto coerente
    vmin = min(x_clean_test.min(), x_noisy_test.min(), x_filtered_test.min())
    vmax = max(x_clean_test.max(), x_noisy_test.max(), x_filtered_test.max())

    def plot_graph_signal(ax, signal, title):
        for i in range(g.N):
            for j in range(i + 1, g.N):
                if A[i, j] > 0:
                    ax.plot([g.coords[i, 0], g.coords[j, 0]], [g.coords[i, 1], g.coords[j, 1]], 'gray', alpha=0.3, zorder=1)
        scatter = ax.scatter(g.coords[:, 0], g.coords[:, 1], c=signal, cmap='coolwarm', vmin=vmin, vmax=vmax, s=120, edgecolors='k', zorder=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        return scatter

    sc1 = plot_graph_signal(axes2[0], x_clean_test, "1) Segnale Pulito (Ground Truth)")
    plot_graph_signal(axes2[1], x_noisy_test, f"2) Segnale Rumoroso (SNR = {snr_in_test:.1f} dB)")
    plot_graph_signal(axes2[2], x_filtered_test, f"3) Ricostruzione Data-Driven (SNR = {snr_out_test:.1f} dB)")

    fig2.subplots_adjust(right=0.88, wspace=0.2)
    cbar_ax = fig2.add_axes([0.91, 0.2, 0.02, 0.6])
    fig2.colorbar(sc1, cax=cbar_ax, label="Ampiezza del Segnale")

    plt.show()