"""
dbscan_min_samples.py
Justificación del mínimo de 3 categorías de DBSCAN (sección 4.3).

Corre DBSCAN sobre los perfiles mensuales de E2 y E3 con los mismos cinco radios del artículo
(percentiles 10 a 50 de las distancias entre categorías) y con distintos mínimos de categorías por grupo.
Muestra, para cada radio, cuántos grupos formó y cuántas categorías quedaron como ruido.

Resultado esperado: con un mínimo de 13, en los dos radios más chicos de E2 todas las categorías
quedan como ruido; con 3 o 5, DBSCAN forma un solo grupo en todos los radios.

Uso (desde la carpeta algoritmos):  python dbscan_min_samples.py
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'simulacion' / 'scripts'))
from comun import escenario, perfiles  # noqa: E402

MESES = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SET', 'OCT', 'NOV', 'DIC']

for e in ['E2', 'E3']:
    s = escenario(e)
    X = perfiles(s[MESES].values.astype(float))
    radios = np.percentile(pdist(X), [10, 20, 30, 40, 50])
    print(f'{e} ({len(X)} categorías)')
    for minimo in [3, 5, 13]:
        res = []
        for r in radios:
            l = DBSCAN(eps=r, min_samples=minimo).fit_predict(X)
            res.append(f'{len(set(l) - {-1})} grupo(s), {int((l == -1).sum())} de ruido')
        print(f'  mínimo {minimo:>2}: ' + ' | '.join(res))
