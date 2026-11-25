"""
fig2_mapa_calor_E2.py
Figura 2: mapa de calor de los perfiles mensuales de las categorías de E2 (2025).

Qué se grafica:
  - E2 = categorías con al menos un trasplante en cada uno de los 12 meses
    (mismo criterio que scripts/comun.py del paquete científico: Meses == 12).
  - Cada fila es una categoría; cada columna, un mes.
  - El color es la proporción del total anual de la categoría que cae en ese mes:
        p(i, m) = x(i, m) / T(i),    T(i) = suma de los 12 meses de la categoría i
    Es la misma representación que usan los algoritmos de clustering.
  - Las filas se ordenan de mayor a menor total anual; el total va entre paréntesis.

Entrada: csv_originales/salida13_organo_tejido_2025_cab2963.csv
Salida:  figuras/fig2_mapa_calor_E2.png (300 dpi)
         figuras/fig2_valores.csv (las proporciones graficadas)

Uso (desde la carpeta algoritmos):  python fig2_mapa_calor_E2.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')                 # genera el archivo sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd

A_MANO = Path(__file__).resolve().parents[1]           # raíz del repositorio
ENTRADA = A_MANO / 'csv_originales' / 'salida13_organo_tejido_2025_cab2963.csv'
SALIDA_PNG = A_MANO / 'figuras' / 'fig2_mapa_calor_E2.png'
SALIDA_CSV = A_MANO / 'figuras' / 'fig2_valores.csv'
MESES = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SET', 'OCT', 'NOV', 'DIC']

# 1) Leer la salida 13 (separador «;»).
tabla = pd.read_csv(ENTRADA, sep=';', encoding='utf-8-sig')
tabla = tabla.rename(columns={tabla.columns[1]: 'Clasificacion'})        # «Clasificación»
tabla['categoria'] = tabla['Clasificacion'].str.split('*', n=1).str[1].str.strip()

# 2) Control: la suma de los 12 meses tiene que dar el TOTAL de cada fila.
assert (tabla[MESES].sum(axis=1) == tabla['TOTAL']).all(), 'Algún TOTAL no coincide con sus meses'
print('Categorías en el archivo:', len(tabla), '| conteos:', tabla['TOTAL'].sum())   # 78 y 8492

# 3) Seleccionar E2: actividad en los 12 meses.
meses_activos = (tabla[MESES] > 0).sum(axis=1)
e2 = tabla[meses_activos == 12].sort_values('TOTAL', ascending=False)
print('Categorías en E2:', len(e2), '| conteos:', e2['TOTAL'].sum())                # 22 y 7830

# 4) Proporción mensual de cada categoría: cada fila suma 1.
proporciones = e2[MESES].div(e2['TOTAL'], axis=0)
proporciones.index = [f"{c} ({t})" for c, t in zip(e2['categoria'], e2['TOTAL'])]
proporciones.to_csv(SALIDA_CSV, sep=';', encoding='utf-8-sig', float_format='%.4f')

# 5) Dibujar.
fig, ax = plt.subplots(figsize=(7.5, 7))
imagen = ax.imshow(proporciones.values, aspect='auto', cmap='viridis')
ax.set_xticks(range(12))
ax.set_xticklabels(MESES, fontsize=8)
ax.set_yticks(range(len(proporciones)))
ax.set_yticklabels(proporciones.index, fontsize=7)
ax.set_xlabel('Mes de 2025 (fecha de desclampeo)')
barra = fig.colorbar(imagen, ax=ax)
barra.set_label('Proporción del total anual de la categoría')
fig.tight_layout()
fig.savefig(SALIDA_PNG, dpi=300)
print('Figura guardada en:', SALIDA_PNG)
