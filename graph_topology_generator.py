import os
import pickle
import libpysal
import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt


class GraphTopology:
    """Gestisce la topologia della rete, la matrice di adiacenza simmetrica,

    il Laplaciano Normalizzato L_norm, la stima spettrale e la base GFT.
    """

    def __init__(self, num_nodes=50, k_neighbors=4, seed=42, test=False):

        self.N = num_nodes
        self.k = k_neighbors
        self.seed = seed
        self.test = test

        self.A_binary = None
        self.W_row_norm = None
        self.L_norm_sparse = None
        self.lambda_max = None
        self.eigenvalues = None
        self.U = None

        if self.test:
            self._build_topology_test()
        else:
            np.random.seed(self.seed)
            self.coords = np.random.rand(self.N, 2) * 100.0
            self._build_topology()

        self._check_connectivity()
        self._compute_normalized_laplacian()
        self._estimate_lambda_max()

    def _build_topology(self):

        w_knn = libpysal.weights.KNN.from_array(self.coords, k=self.k)
        print(w_knn.full()[0])
        A_dense = w_knn.full()[0]
        A_sym = np.maximum(A_dense, A_dense.T)
        np.fill_diagonal(A_sym, 0)
        self.A_binary = csr_matrix(A_sym)

        w_knn.transform = 'R'
        self.W_row_norm = csr_matrix(w_knn.full()[0])

    def _build_topology_test(self):
        self.N = 3
        self.coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        A_sym = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        self.A_binary = csr_matrix(A_sym)
        self.W_row_norm = csr_matrix(A_sym / 2.0)

    def _check_connectivity(self):

        n_comp, _ = connected_components(self.A_binary, directed=False)
        if n_comp > 1:
            print(f"⚠️ Warning: Il grafo ha {n_comp} componenti disconnesse.")

    def _compute_normalized_laplacian(self):

        degrees = np.array(self.A_binary.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(degrees, dtype=float)
        np.power(degrees, -0.5, where=degrees > 0, out=d_inv_sqrt)
        D_inv_sqrt = diags(d_inv_sqrt, format='csr')
        I_sparse = eye(self.N, format='csr')
        self.L_norm_sparse = I_sparse - D_inv_sqrt.dot(self.A_binary).dot(D_inv_sqrt)

    def _estimate_lambda_max(self, max_iter=300, tol=1e-5):
        np.random.seed(self.seed)
        x = np.random.randn(self.N)
        x = x / np.linalg.norm(x)
        lambda_old = 0.0

        for _ in range(max_iter):
            x_next = self.L_norm_sparse.dot(x)
            norm = np.linalg.norm(x_next)
            if norm == 0:
                break
            x = x_next / norm
            lambda_curr = float(x.T.dot(self.L_norm_sparse.dot(x)))
            if abs(lambda_curr - lambda_old) < tol:
                break
            lambda_old = lambda_curr

        self.lambda_max = min(lambda_curr, 2.0)

    def compute_fourier_basis(self):

        """Calcola autovalori e autovettori di L_norm (Base GFT)."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.L_norm_sparse.toarray())
        idx = np.argsort(eigenvalues)
        self.eigenvalues = np.clip(eigenvalues[idx], 0.0, 2.0)
        self.U = eigenvectors[:, idx]
        return self.eigenvalues, self.U

    def save_topology(self, filepath='./resources/graphs/', name='graph_topology.pkl'):

        data = {
            'N': self.N,
            'k': self.k,
            'seed': self.seed,
            'coords': self.coords,
            'W_row_norm': self.W_row_norm,
            'A_binary': self.A_binary,
            'L_norm_sparse': self.L_norm_sparse,
            'lambda_max': self.lambda_max,
            'eigenvalues': self.eigenvalues,
            'U': self.U,
        }
        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, name)
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Topologia salvata in: '{full_path}'")
   
    def plot_graph(self, show_labels=False, node_size=40):
        A = self.A_binary.tocoo()

        plt.figure(figsize=(7, 7))

        # Disegna archi (solo i<j per non duplicare, dato che A è simmetrica)
        for i, j, w in zip(A.row, A.col, A.data):
            if i < j and w != 0:
                xi, yi = self.coords[i]
                xj, yj = self.coords[j]
                plt.plot([xi, xj], [yi, yj], color="gray", linewidth=0.8, alpha=0.7)

        # Disegna nodi
        plt.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            s=node_size,
            c="tab:blue",
            edgecolors="black",
            linewidths=0.4,
            zorder=3
        )

        if show_labels:
            for n, (x, y) in enumerate(self.coords):
                plt.text(x + 0.6, y + 0.6, str(n), fontsize=8)

        plt.title(f"Grafo KNN (N={self.N}, k={self.k})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    topology = GraphTopology(num_nodes=50, k_neighbors=4, seed=42)
    topology.plot_graph(show_labels=True)
