# Progettazione e analisi comparativa di filtri polinomiali ai minimi quadrati per il denoising di segnali su grafo

Codice per la tesina del corso di **Signal Processing and Optimization for Big Data** (Corso di Laurea Magistrale in Ingegneria Informatica e Robotica — Università degli Studi di Perugia).

---

## Panoramica del progetto

Il progetto implementa e confronta algoritmi di filtraggio polinomiale su grafo per la rimozione di rumore bianco gaussiano additivo (AWGN) da segnali a banda limitata definiti su domini irregolari (grafi $k$-NN).

Vengono analizzate e confrontate tre strategie di stima dei coefficienti del filtro $\mathbf{h} \in \mathbb{R}^{K+1}$:
1. **LS spettrale "Oracolo":** Benchmark teorico basato sulla conoscenza esatta della frequenza di taglio $\lambda_c$.
2. **LS spettrale da energia cumulativa:** Stima euristica basata sulla soglia al $75\%$ dello spettro GFT degradato.
3. **Data-Driven (Krylov Subspace Regression):** Apprendimento supervisionato dei pesi tramite Ridge Regression/Tikhonov direttamente nello spazio dei nodi.

---

## Struttura del repository

```text
.
├── graph_topology_generator.py   # Costruzione topologia k-NN, GSO (L_norm) e base GFT
├── graph_signal_generator.py     # Generatore di segnali a banda limitata e disturbo AWGN
├── graph_filter_ls.py            # Modulo filtri: Least Squares Spettrale e Data-Driven
├── main_energy.py                # Pipeline di test per i metodi spettrali (Oracolo vs Energia)
├── main_data_driven.py           # Pipeline di test per il metodo Data-Driven (Train/Test)
├── grid_search_energy.py         # Grid search parametrica (K x SNR_in) per metodi spettrali
├── grid_search_datadriven.py     # Grid search parametrica (K x SNR_in) per il metodo Data-Driven
├── requirements.txt              # Dipendenze Python necessarie
└── README.md
```

### Creazione e attivazione dell'ambiente virtuale

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
