import numpy as np
from graph_topology_generator import GraphTopology
import matplotlib.pyplot as plt


class GraphFilterLS:

    def __init__(self, topology: GraphTopology, order: int = 5, reg_param: float = 1e-4):

        self.topology = topology
        self.K = order
        self.mu = reg_param
        self.h = None

    def fit_spectral(self, target_filter_fn):
        """Metodo 1 (Spettrale): Calibrazione dei coefficienti h del filtro polinomiale

        per approssimare una risposta spettrale desiderata g(lambda).

        Problema di ottimizzazione risolto (Ridge Regression / Tikhonov):
            min_h ||V * h - g_target||_2^2 + mu * ||h||_2^2
        """

        lambdas = self.topology.eigenvalues  
        g_target = np.array([target_filter_fn(lam) for lam in lambdas])  

        V = np.column_stack([lambdas**k for k in range(self.K + 1)]) 
        I_reg = np.eye(self.K + 1)  
        A_ls = V.T @ V + self.mu * I_reg  
        b_ls = V.T @ g_target  
        self.h = np.linalg.solve(A_ls, b_ls) 

        return self.h

    def fit_data_driven(self, x_in: np.ndarray, y_target: np.ndarray):
        """Metodo 2 (Data-Driven): Stima dei coefficienti h del filtro polinomiale

        nello spazio dei vertici tramite regressione sui sottospazi di Krylov.

        Problema di ottimizzazione supervisionato (Tikhonov / Ridge Regression):
            min_h ||X * h - y_target||_2^2 + mu * ||h||_2^2
            dove X e' la matrice delle risposte diffuse a k-hop: X = [x_in, L*x_in, ..., L^K*x_in].
        """

        X = self._build_krylov_matrix(x_in) 
        I_reg = np.eye(self.K + 1) 
        A_ls = X.T @ X + self.mu * I_reg  
        b_ls = X.T @ y_target  
        self.h = np.linalg.solve(A_ls, b_ls)  

        return self.h
    

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Applica direttamente il filtro polinomiale: y = H(L)x = sum_{k=0}^K h_k * (L^k x).
        y = H(L)x = \sum_{k=0}^K h_k L^k x = h_0 x + h_1 L x + h_2 L^2 x + ... + h_K L^K x
        """

        L = self.topology.L_norm_sparse


        curr = x.copy()
        y = self.h[0] * curr

        for k in range(1, self.K + 1):
            curr = L.dot(curr)
            y += self.h[k] * curr

        return y


    def _build_krylov_matrix(self, x: np.ndarray) -> np.ndarray:

        """Calcola ricorsivamente le diffusioni x^(k) = L^k * x."""

        L = self.topology.L_norm_sparse
        shifts = [x]
        curr = x.copy()

        for _ in range(self.K):
            curr = L.dot(curr)
            shifts.append(curr)

        return np.column_stack(shifts)

    def plot_frequency_response(
        self,
        target_filter_fn=None,
        title: str = None,
        num_points: int = 300,
    ):
        """Visualizza la risposta in frequenza polinomiale h(lambda)."""
        if self.h is None:
            raise ValueError(
                "I coefficienti h non sono stati ancora calcolati. Esegui prima fit_spectral o fit_data_driven."
            )

        lambdas = self.topology.eigenvalues
        lambda_max = np.max(lambdas)

        # Griglia continua per visualizzare la curva del polinomio
        lam_grid = np.linspace(0, lambda_max, num_points)

        # Usiamo h[::-1] perche np.polyval si aspetta i coefficienti in ordine decrescente [h_K, ..., h_0]
        h_continuous = np.polyval(self.h[::-1], lam_grid)
        h_discrete = np.polyval(self.h[::-1], lambdas)

        plt.figure(figsize=(9, 5))

        plt.plot(
            lam_grid,
            h_continuous,
            label=f"Filtro Polinomiale (K={self.K})",
            color="navy",
            linewidth=2,
        )

        if target_filter_fn is not None:
            g_target_grid = np.array([target_filter_fn(lam) for lam in lam_grid])
            plt.plot(
                lam_grid,
                g_target_grid,
                "r--",
                label=r"Target Ideale $g(\lambda)$",
                alpha=0.8,
            )

        plt.scatter(
            lambdas,
            h_discrete,
            color="crimson",
            s=20,
            zorder=3,
            label=r"Autovalori $\lambda_i$",
        )

        plt.axhline(0, color="gray", linestyle=":", alpha=0.6)
        plt.axhline(1, color="gray", linestyle=":", alpha=0.6)

        # Imposta il titolo passato o usa quello di default
        plot_title = (
            title
            if title is not None
            else rf"Risposta in Frequenza del Filtro: $h(\lambda) = \sum_{{k=0}}^{{K}} h_k \lambda^k$"
        )
        plt.title(plot_title, fontsize=12)

        plt.xlabel(r"Frequenza sul Grafo (Autovalori $\lambda$)", fontsize=11)
        plt.ylabel(r"Guadagno $h(\lambda)$", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()