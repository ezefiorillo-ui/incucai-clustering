"""
ic_controles_positivos.py
Intervalos de confianza del 95 % de las secciones 5.2 y 5.3 (controles positivos, dos grupos, δ = 0,3).

  - ARI medio: media ± 1,96 · desvío estándar / raíz(100), sobre las 100 repeticiones
    (simulacion/resultados/control_positivo.csv).
  - Detección: intervalo de Wilson, ya calculado en simulacion/resultados/tabla_potencia.csv
    (columnas IC95_inf e IC95_sup).

Uso (desde la carpeta algoritmos):  python ic_controles_positivos.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / 'simulacion' / 'resultados'

cp = pd.read_csv(RES / 'control_positivo.csv')
g = cp[(cp['G'] == 2) & (cp['delta'] == 0.3)].groupby(['escenario', 'algoritmo'])['ARI_elegido']
ari = pd.DataFrame({'media': g.mean(), 'desvio': g.std(), 'n': g.count()})
ari['IC95_inf'] = ari['media'] - 1.96 * ari['desvio'] / np.sqrt(ari['n'])
ari['IC95_sup'] = ari['media'] + 1.96 * ari['desvio'] / np.sqrt(ari['n'])
print('ARI medio, dos grupos, delta = 0,3')
print(ari.round(3).to_string())

pot = pd.read_csv(RES / 'tabla_potencia.csv')
sel = pot[(pot['ruido'] == 'N0') & (pot['G'] == 2) & (pot['delta'] == 0.3)
          & pot['procedimiento'].str.match(r'^(A  máx\. silhouette K-Means|R  reducción de dispersión K-Means k=2|G  gap)')]
print('\nDetección, dos grupos, delta = 0,3 (intervalo de Wilson)')
print(sel[['escenario', 'procedimiento', 'deteccion', 'IC95_inf', 'IC95_sup']].round(3).to_string(index=False))
