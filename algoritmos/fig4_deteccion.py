"""
fig4_deteccion.py
Figura 4: frecuencia con que cada criterio detecta estructura en el control positivo.

Qué se grafica:
  - Mismos datos simulados que la Figura 3 (perfiles sinusoidales con intensidad δ),
    con K-Means y 100 réplicas por condición, bajo el modelo nulo N0 (multinomial).
  - «Detectar» = el estadístico de la réplica supera el percentil 95 de su distribución
    bajo el nulo (sin grupos). Para el gap: elegir al menos dos grupos.
  - Eje y: proporción de las 100 réplicas en que el criterio detectó estructura.
    Con δ = 0 (sin grupos) debería rondar 0,05 (línea punteada): son falsos positivos.
  - Bandas: intervalo de confianza del 95 % (Wilson) de cada proporción.
  - Paneles: escenario E2 / E3 (filas) y G = 2 / 3 (columnas).

Criterios (columna «procedimiento» de la tabla):
  A  máximo silhouette de K-Means, k = 2 a 6
  B  silhouette de K-Means con el k verdadero (solo diagnóstico: en datos reales no se conoce)
  R  reducción de dispersión de K-Means, k = 2:   R_k = 1 − W_k / W_1
  G  gap statistic
  Z  reducción de dispersión sobre residuos de Pearson, k = 2

Entrada: resultados_simulacion/tabla_potencia.csv
         (copia de resultados/tabla_potencia.csv del paquete científico, generada por
          scripts/s06_deteccion.py y resumida por scripts/s07_consolidar_deteccion.py)
Salida:  figuras/fig4_deteccion.png (300 dpi)

Uso (desde la carpeta algoritmos):  python fig4_deteccion.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

# Marcas de los ejes: en x, exactamente los δ simulados; en y, cada 0,1. Con coma decimal.
DELTAS = [0, 0.2, 0.3, 0.5, 0.8]
MARCAS_Y = [i / 10 for i in range(11)]
COMA = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ','))

A_MANO = Path(__file__).resolve().parents[1]
ENTRADA = A_MANO / 'resultados_simulacion' / 'tabla_potencia.csv'
SALIDA_PNG = A_MANO / 'figuras' / 'fig4_deteccion.png'

# (texto exacto del procedimiento en la tabla, etiqueta, color, estilo de línea)
CRITERIOS = [('A  máx. silhouette K-Means (k=2–6)', 'A: máx. silhouette', 'tab:red', '-'),
             ('B  silhouette k verdadero (solo diagnóstico)', 'B: silhouette, k verdadero', 'tab:red', ':'),
             ('R  reducción de dispersión K-Means k=2', 'R: reducción de dispersión', 'tab:blue', '-'),
             ('G  gap statistic (referencia = nulo)', 'G: gap', 'tab:green', '-'),
             ('Z  reducción dispersión residuos k=2', 'Z: residuos de Pearson', 'tab:gray', '--')]

tabla = pd.read_csv(ENTRADA, encoding='utf-8')
tabla = tabla[tabla.ruido == 'N0']
for texto, *_ in CRITERIOS:
    assert (tabla.procedimiento == texto).any(), f'No se encontró el criterio: {texto}'

# Control: valores citados en el paper (E2, G=2, δ=0,3): B 0/100 y R 96/100.
control = tabla[(tabla.escenario == 'E2') & (tabla.G == 2) & (tabla.delta == 0.3)]
print(control[['procedimiento', 'deteccion']].to_string(index=False))

fig, ejes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
for i, escenario in enumerate(['E2', 'E3']):
    for j, G in enumerate([2, 3]):
        ax = ejes[i, j]
        sub = tabla[(tabla.escenario == escenario) & (tabla.G == G)]
        for texto, etiqueta, color, estilo in CRITERIOS:
            serie = sub[sub.procedimiento == texto].sort_values('delta')
            ax.plot(serie.delta, serie.deteccion, ls=estilo, marker='o', ms=3, color=color, label=etiqueta)
            ax.fill_between(serie.delta, serie.IC95_inf, serie.IC95_sup, color=color, alpha=0.1)
        ax.axhline(0.05, color='black', ls='--', lw=0.7)
        ax.axvline(0.3, color='black', ls=':', lw=0.7)  # δ = 0,3, el caso citado en el texto
        ax.set_title(f'{escenario}, G = {G}', fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(DELTAS)
        ax.set_yticks(MARCAS_Y)
        ax.xaxis.set_major_formatter(COMA)
        ax.yaxis.set_major_formatter(COMA)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if i == 1:
            ax.set_xlabel('Intensidad de la separación δ')
fig.supylabel('Proporción de simulaciones con detección', fontsize=10)
manejadores, etiquetas = ejes[0, 0].get_legend_handles_labels()
fig.legend(manejadores, etiquetas, loc='lower center', ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(SALIDA_PNG, dpi=300)
print('Figura guardada en:', SALIDA_PNG)
