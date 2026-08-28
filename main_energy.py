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


if __name__ == '__main__':

    
    # =========================================================================
    # SEZIONE 0: TOPOLOGIA DEL GRAFO E GENERAZIONE DATI
    # =========================================================================
    K_order = 6       # Ordine polinomiale comune
    cutoff_idx = 5    # Numero di modi a bassa frequenza usati per la generazione

    g = GraphTopology(num_nodes=50, k_neighbors=4, seed=42)
    sig_gen = GraphSignalGenerator(g)

    # Generazione segnale pulito e aggiunta di AWGN (8 dB)
    x_clean = sig_gen.generate_bandlimited_signal(cutoff_idx=cutoff_idx, seed=10)
    x_noisy, _ = sig_gen.add_noise(x_clean, snr_db=8.0, seed=20)
    
    mse_noisy = np.mean((x_noisy - x_clean) ** 2)
    snr_in = compute_snr(x_clean, x_noisy)

    # =========================================================================
    # SEZIONE 1: FILTRO SPETTRALE ORACOLO (Conoscenza a Priori)
    # =========================================================================
    lambda_cutoff_oracle = g.eigenvalues[cutoff_idx - 1]
    
    def ideal_lowpass_oracle(lam):
        return 1.0 if lam <= lambda_cutoff_oracle else 0.0

    filter_oracle = GraphFilterLS(g, order=K_order, reg_param=1e-5)
    filter_oracle.fit_spectral(ideal_lowpass_oracle)
    filter_oracle.plot_frequency_response(target_filter_fn=ideal_lowpass_oracle, title="Filtro Oracolo")
    x_filtered_oracle = filter_oracle.filter(x_noisy)

    mse_oracle = np.mean((x_filtered_oracle - x_clean) ** 2)
    snr_out_oracle = compute_snr(x_clean, x_filtered_oracle)

    # =========================================================================
    # SEZIONE 2: FILTRO SPETTRALE DA ENERGIA CUMULATIVA (Stima Empirica)
    # =========================================================================
    gft_energy = (g.U.T @ x_noisy) ** 2
    cum_energy = np.cumsum(gft_energy) / np.sum(gft_energy)
    cutoff_idx_energy = np.searchsorted(cum_energy, 0.75)
    lambda_cutoff_energy = g.eigenvalues[cutoff_idx_energy]

    def ideal_lowpass_energy(lam):
        return 1.0 if lam <= lambda_cutoff_energy else 0.0

    filter_energy = GraphFilterLS(g, order=K_order, reg_param=1e-5)
    filter_energy.fit_spectral(ideal_lowpass_energy)
    filter_energy.plot_frequency_response(target_filter_fn=ideal_lowpass_energy, title="Filtro Energia")
    x_filtered_energy = filter_energy.filter(x_noisy)

    mse_energy = np.mean((x_filtered_energy - x_clean) ** 2)
    snr_out_energy = compute_snr(x_clean, x_filtered_energy)

    # =========================================================================
    # SEZIONE 3: REPORT TERMINALE
    # =========================================================================
    print("=" * 70)
    print("CONFRONTO FILTRI SPETTRALI SU GRAFO (LEAST SQUARES)")
    print("=" * 70)
    print(f"Taglio Oracolo:  idx = {cutoff_idx:2d} | lambda_c = {lambda_cutoff_oracle:.4f}")
    print(f"Taglio Energia:  idx = {cutoff_idx_energy:2d} | lambda_c = {lambda_cutoff_energy:.4f}")
    print("-" * 70)
    print(f"Segnale Rumoroso:         MSE = {mse_noisy:.4f} | SNR = {snr_in:.2f} dB")
    print(f"1) LS Spettrale (Oracolo):MSE = {mse_oracle:.4f} | SNR = {snr_out_oracle:.2f} dB (+{snr_out_oracle - snr_in:.2f} dB)")
    print(f"2) LS Spettrale (Energia):MSE = {mse_energy:.4f} | SNR = {snr_out_energy:.2f} dB (+{snr_out_energy - snr_in:.2f} dB)")
    print("=" * 70)

    # =========================================================================
    # SEZIONE 4: FIGURA 1 - ANALISI QUANTITATIVA E SPETTRALE (1x2)
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5))
    nodes = np.arange(g.N)

    # [1.1] Segnali sui Nodi
    ax = axes1[0]
    ax.plot(nodes, x_clean, 'k-', linewidth=2, label='Clean (Ground Truth)')
    ax.plot(nodes, x_noisy, 'r.', alpha=0.4, label=f'Noisy ({snr_in:.1f} dB)')
    ax.plot(nodes, x_filtered_oracle, 'b.-', alpha=0.7, label=f'Oracolo ({snr_out_oracle:.1f} dB)')
    ax.plot(nodes, x_filtered_energy, 'm.--', alpha=0.7, label=f'Energia ({snr_out_energy:.1f} dB)')
    ax.set_title("Segnali Ricostruiti sui Vertici", fontsize=11)
    ax.set_xlabel("Indice del Nodo")
    ax.set_ylabel("Ampiezza")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    # [1.2] Spettro GFT
    ax = axes1[1]
    gft_clean = np.abs(g.U.T @ x_clean)
    gft_noisy = np.abs(g.U.T @ x_noisy)
    gft_oracle = np.abs(g.U.T @ x_filtered_oracle)
    gft_energy_spec = np.abs(g.U.T @ x_filtered_energy)

    ax.stem(nodes, gft_noisy, linefmt='r:', markerfmt='rx', basefmt='k-', label='Rumoroso')
    ax.stem(nodes, gft_oracle, linefmt='b-', markerfmt='bo', basefmt='k-', label='Oracolo')
    ax.stem(nodes, gft_energy_spec, linefmt='m:', markerfmt='m^', basefmt='k-', label='Energia')
    ax.stem(nodes, gft_clean, linefmt='k--', markerfmt='k.', basefmt='k-', label='Clean')
    ax.set_title(r"Spettro GFT: $|\hat{x}_i|$", fontsize=11)
    ax.set_xlabel("Indice di Frequenza ($i$)")
    ax.set_ylabel("Ampiezza Spettrale")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    fig1.tight_layout()

    # =========================================================================
    # SEZIONE 5: FIGURA 2 - CONFRONTO TOPOLOGICO SUL GRAFO (2x2)
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    A = g.A_binary.toarray()

    # Normalizzazione comune per confronto visivo coerente
    vmin = min(x_clean.min(), x_noisy.min(), x_filtered_oracle.min(), x_filtered_energy.min())
    vmax = max(x_clean.max(), x_noisy.max(), x_filtered_oracle.max(), x_filtered_energy.max())

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

    # Prima Riga: Ground Truth e Segnale Rumoroso
    sc1 = plot_graph_signal(axes2[0, 0], x_clean, "Segnale Pulito (Ground Truth)")
    plot_graph_signal(axes2[0, 1], x_noisy, f"Segnale Rumoroso (SNR = {snr_in:.1f} dB)")

    # Seconda Riga: Ricostruzioni Spettrali
    plot_graph_signal(axes2[1, 0], x_filtered_oracle, f"1) LS Oracolo (SNR = {snr_out_oracle:.1f} dB)")
    plot_graph_signal(axes2[1, 1], x_filtered_energy, f"2) LS Energia (SNR = {snr_out_energy:.1f} dB)")

    fig2.subplots_adjust(right=0.88, hspace=0.25, wspace=0.2)
    cbar_ax = fig2.add_axes([0.91, 0.15, 0.02, 0.7])
    fig2.colorbar(sc1, cax=cbar_ax, label="Ampiezza del Segnale")

    plt.show()