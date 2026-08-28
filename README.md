# Progettazione e Analisi Comparativa di Filtri Polinomiali ai Minimi Quadrati per il Denoising di Segnali su Grafo

Codice per la tesina del corso di **Signal Processing and Optimization for Big Data** (Corso di Laurea Magistrale in Ingegneria Informatica e Robotica — Università degli Studi di Perugia).

---

## Panoramica del Progetto

Il progetto implementa e confronta algoritmi di filtraggio polinomiale su grafo per la rimozione di rumore bianco gaussiano additivo (AWGN) da segnali a banda limitata definiti su domini irregolari (grafi $k$-NN).

Vengono analizzate e confrontate tre strategie di stima dei coefficienti del filtro $\mathbf{h} \in \mathbb{R}^{K+1}$:
1. **LS Spettrale "Oracolo":** Benchmark teorico basato sulla conoscenza esatta della frequenza di taglio $\lambda_c$.
2. **LS Spettrale da Energia Cumulativa:** Stima euristica data-free basata sulla soglia al $75\%$ dello spettro GFT degradato.
3. **Data-Driven (Krylov Subspace Regression):** Apprendimento supervisionato dei pesi tramite Ridge Regression/Tikhonov direttamente nello spazio dei nodi, validato su istanze *out-of-sample*.

---

## Struttura della Repository

```text
.
├── graph_topology_generator.py   # Costruzione topologia k-NN, GSO (L_norm) e base GFT
├── graph_signal_generator.py     # Generatore di segnali a banda limitata e disturbo AWGN
├── graph_filter_ls.py            # Modulo filtri: Least Squares Spettrale e Data-Driven (Krylov)
├── main_energy.py                # Pipeline di test per i metodi spettrali (Oracolo vs Energia)
├── main_data_driven.py           # Pipeline di test per il metodo Data-Driven (Train/Test Out-of-Sample)
├── grid_search_energy.py         # Grid search parametrica (K x SNR_in) per metodi spettrali
├── grid_search_datadriven.py     # Grid search parametrica (K x SNR_in) per il metodo Data-Driven
├── requirements.txt              # Dipendenze Python necessarie
└── README.md

### Creazione e Attivazione dell'Ambiente Virtuale

* **Su Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **Su Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

* **Su Windows (Prompt dei comandi / CMD):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```


```text
pip install --upgrade pip
pip install -r requirements.txt
