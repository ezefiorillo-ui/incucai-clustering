"""
holm_y_magnitud_Z.py
Números de la sección 5.4 que no salen directamente de una tabla:

  1. Corrección de Holm sobre la familia de 22 pruebas (11 procedimientos × E2 y E3, frente a N0).
  2. Magnitud de Z en E2: valor real, mediana y percentil 95 de las 1000 simulaciones de N0.

Entrada: simulacion/resultados/reales_todos_estadisticos.csv  (p empíricos de los datos reales)
         simulacion/resultados/det_null_E2_N0.csv             (1000 simulaciones de N0 para E2)

Uso (desde la carpeta algoritmos):  python holm_y_magnitud_Z.py
"""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parents[1] / 'simulacion' / 'resultados'

# 1. Holm: se ordenan los p de menor a mayor; el i-ésimo (desde 0) se multiplica por (m - i)
#    y se toma el máximo acumulado, para que los p ajustados no bajen.
r = pd.read_csv(RES / 'reales_todos_estadisticos.csv')
fam = r[(r.ruido == 'N0') & r.p_empirico.notna()].sort_values('p_empirico').reset_index(drop=True)
m = len(fam)
ajustados, maximo = [], 0.0
for i, p in enumerate(fam.p_empirico):
    maximo = max(maximo, min(1.0, (m - i) * p))
    ajustados.append(maximo)
fam['p_Holm'] = ajustados
print(f'Familia de Holm: {m} pruebas (11 procedimientos x E2 y E3, frente a N0)\n')
print(fam[['escenario', 'procedimiento', 'p_empirico', 'p_Holm']].round(4).to_string(index=False))

# 2. Magnitud de Z en E2 (reducción de dispersión sobre residuos de Pearson, k = 2)
nulo = pd.read_csv(RES / 'det_null_E2_N0.csv')['R_Z_k2']
real = r[(r.escenario == 'E2') & (r.ruido == 'N0') & r.procedimiento.str.startswith('Z  reducc')].valor.iloc[0]
print(f'\nZ en E2: real = {real:.3f} | N0: mediana = {nulo.median():.3f}, percentil 95 = {nulo.quantile(0.95):.3f} '
      f'({len(nulo)} simulaciones)')
