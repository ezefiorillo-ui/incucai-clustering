"""
fig3_ARI_algoritmos.py
Figura 3: recuperación de grupos conocidos (ARI) por algoritmo en el control positivo.

Qué se grafica:
  - Control positivo: se asignan al azar las categorías reales a G grupos (G = 2 o 3)
    y cada grupo recibe un perfil mensual distinto,
        perfil_g(m) ∝ p0(m) · [1 + δ · sen(2πm/12 + 2πg/G)],
    donde δ es la intensidad de la separación (δ = 0: sin grupos).
    Con ese perfil y los totales reales se simulan conteos multinomiales (100 réplicas).
  - A cada réplica se le aplican K-Means, Ward, GMM y DBSCAN; el número de grupos se elige
    con el máximo silhouette (k = 2 a 6).
  - Eje y: ARI medio entre la partición elegida y los grupos verdaderos
    (1 = recuperación perfecta; 0 = como al azar).
  - Paneles: escenario E2 / E3 (filas) y G = 2 / 3 (columnas).

Entrada: resultados_simulacion/tabla_control_positivo.csv
         (copia de resultados/tabla_control_positivo.csv del paquete científico, generada por
          scripts/s03_control_positivo.py y resumida por scripts/s04_consolidar_control_positivo.py)
Salida:  figuras/fig3_ARI_algoritmos.png (300 dpi)

Uso (desde la carpeta algoritmos):  python fig3_ARI_algoritmos.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

# Marcas de los ejes: en x, exactamente los δ simulados; en y, cada 0,1. Con coma decimal.
DELTAS = [0, 0.1, 0.2, 0.3, 0.5, 0.8]
MARCAS_Y = [i / 10 for i in range(11)]
COMA = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))

A_MANO = Path(__file__).resolve().parents[1]
ENTRADA = A_MANO / 'resultados_simulacion' / 'tabla_control_positivo.csv'
SALIDA_PNG = A_MANO / 'figuras' / 'fig3_ARI_algoritmos.png'

ALGORITMOS = [('KMeans', 'K-Means', 'tab:blue', 'o'),
              ('Ward', 'Ward', 'tab:orange', 's'),
              ('GMM', 'GMM', 'tab:green', '^'),
              ('DBSCAN', 'DBSCAN', 'tab:red', 'D')]

tabla = pd.read_csv(ENTRADA)

# Control: valores citados en el paper (E2, G=2, δ=0,3).
control = tabla[(tabla.escenario == 'E2') & (tabla.G == 2) & (tabla.delta == 0.3)]
print(control[['algoritmo', 'ARI_elegido_media']].to_string(index=False))

fig, ejes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=True, sharey=True)
for i, escenario in enumerate(['E2', 'E3']):
    for j, G in enumerate([2, 3]):
        ax = ejes[i, j]
        sub = tabla[(tabla.escenario == escenario) & (tabla.G == G)]
        for clave, etiqueta, color, marcador in ALGORITMOS:
            serie = sub[sub.algoritmo == clave].sort_values('delta')
            ax.plot(serie.delta, serie.ARI_elegido_media, marker=marcador, ms=4,
                    color=color, label=etiqueta)
        ax.set_title(f'{escenario}, G = {G}', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(DELTAS)
        ax.set_yticks(MARCAS_Y)
        ax.xaxis.set_major_formatter(COMA)
        ax.yaxis.set_major_formatter(COMA)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if i == 1:
            ax.set_xlabel('Intensidad de la separación δ')
        if j == 0:
            ax.set_ylabel('ARI medio')
ejes[0, 0].legend(fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(SALIDA_PNG, dpi=300)
print('Figura guardada en:', SALIDA_PNG)
