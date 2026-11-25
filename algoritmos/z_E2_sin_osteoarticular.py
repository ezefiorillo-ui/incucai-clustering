"""
z_E2_sin_osteoarticular.py
Comprobación de la sección 5.4: se repite Z (reducción de dispersión sobre residuos de Pearson,
K-Means con k = 2) en E2 sin la categoría SISTEMA OSTEOARTICULAR y se compara con el modelo nulo N0.

  - E2 sin SISTEMA OSTEOARTICULAR: 21 categorías.
  - Modelo nulo N0: multinomial, con el total de cada categoría y el perfil agregado; B = 1000.
  - p empírico = (1 + cantidad de simulaciones con valor >= al real) / (B + 1).

Uso (desde la carpeta algoritmos):  python z_E2_sin_osteoarticular.py
Tarda unos segundos. Resultado esperado: Z = 0,1835 y p = 0,326.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / 'csv_originales' / 'salida13_organo_tejido_2025_cab2963.csv'
MESES = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SET', 'OCT', 'NOV', 'DIC']
B = 1000
SEMILLA = 20261017

tabla = pd.read_csv(ENTRADA, sep=';', encoding='utf-8-sig')
tabla['categoria'] = tabla[tabla.columns[1]].str.split('*', n=1).str[1].str.strip()
e2 = tabla[(tabla[MESES] > 0).sum(axis=1) == 12]
sin_osteo = e2[e2['categoria'] != 'SISTEMA OSTEOARTICULAR']
C = sin_osteo[MESES].to_numpy(float)
totales = C.sum(axis=1).astype(int)
perfil = C.sum(axis=0) / C.sum()


def z(conteos):
    """Reducción de dispersión de K-Means (k = 2) sobre los residuos de Pearson (ecuación 3)."""
    esperado = np.outer(conteos.sum(axis=1), conteos.sum(axis=0) / conteos.sum())
    r = (conteos - esperado) / np.sqrt(esperado)
    return 1 - KMeans(2, random_state=42, n_init=10).fit(r).inertia_ / ((r - r.mean(axis=0)) ** 2).sum()


real = z(C)
simulados = []
for b in range(B):
    rng = np.random.default_rng([SEMILLA, 900, b])
    simulados.append(z(np.vstack([rng.multinomial(t, perfil) for t in totales]).astype(float)))
simulados = np.array(simulados)
p = (1 + (simulados >= real).sum()) / (B + 1)
print(f'E2 sin SISTEMA OSTEOARTICULAR: {len(sin_osteo)} categorías')
print(f'Z = {real:.4f} | p empírico frente a N0 (B = {B}) = {p:.3f}')
