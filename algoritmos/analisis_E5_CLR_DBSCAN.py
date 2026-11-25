"""
analisis_E5_CLR_DBSCAN.py
Análisis pedidos por Roxana en la revisión del 18/11/2026:
  - obs. 11: resultados completos de E5 (E3 sin SISTEMA OSTEOARTICULAR) con los cuatro algoritmos;
  - obs. 3:  sensibilidad con la transformación CLR (centered log-ratio) para datos composicionales;
  - obs. 7:  cuántas categorías marca DBSCAN como ruido en los datos reales.

Qué hace, para cada escenario (E2, E3, E5) y cada representación (proporciones y CLR):
  1. Calcula los mismos criterios que el paper sobre los datos reales:
       - silhouette máximo de K-Means, Ward y GMM (k = 2 a 6) y de DBSCAN (5 radios);
       - reducción de dispersión R2 = 1 - W2/W1 (K-Means, k = 2);
       - Z = reducción de dispersión sobre residuos de Pearson (no depende de la representación);
       - gap (k elegido, con el modelo nulo como referencia).
  2. Genera B = 1000 conjuntos con el modelo nulo N0 (multinomial, mismo total de cada categoría,
     todas con el perfil agregado) y calcula los mismos criterios.
  3. p empírico = (1 + cantidad de simulaciones con valor >= al real) / (B + 1)   (ecuación 6 del paper).

CLR: clr(x)_m = log(x_m) - promedio de los 12 log(x). Como el logaritmo de 0 no existe, a los conteos
se les suma 0,5 antes de calcular las proporciones (en E2 no hay ceros; en E3 y E5 sí).

Es el mismo procedimiento que el paquete científico (scripts/comun.py, s02_nulos.py y s06_deteccion.py):
mismos algoritmos, parámetros, grilla de k y regla de selección (máximo silhouette).

Entrada: csv_originales/salida13_organo_tejido_2025_cab2963.csv
Salida:  resultados_nuevos/tabla_E5_CLR.csv     (valor real y p empírico de cada criterio)
         resultados_nuevos/dbscan_ruido.csv     (grupos y categorías de ruido de DBSCAN en cada radio)
         resultados_nuevos/nulos_<E>_<rep>.csv  (las 1000 simulaciones, para poder revisarlas)

Uso (desde la carpeta algoritmos):  python analisis_E5_CLR_DBSCAN.py
Tarda unos minutos por escenario; va mostrando el avance.
Para una prueba rápida:            python analisis_E5_CLR_DBSCAN.py 20
"""
from pathlib import Path
import sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
warnings.filterwarnings('ignore')

A_MANO = Path(__file__).resolve().parents[1]
ENTRADA = A_MANO / 'csv_originales' / 'salida13_organo_tejido_2025_cab2963.csv'
SALIDA = Path(sys.argv[2]) if len(sys.argv) > 2 else A_MANO / 'resultados_nuevos'
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEMILLA = 20261017                      # la misma del paquete científico
MESES = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SET', 'OCT', 'NOV', 'DIC']
KS = range(2, 7)
REPRESENTACIONES = ['proporciones', 'CLR']

# ---------------------------------------------------------------- datos
tabla = pd.read_csv(ENTRADA, sep=';', encoding='utf-8-sig')
tabla = tabla.rename(columns={tabla.columns[1]: 'Clasificacion'})
tabla['categoria'] = tabla['Clasificacion'].str.split('*', n=1).str[1].str.strip()
assert (tabla[MESES].sum(axis=1) == tabla['TOTAL']).all()
print('Categorías:', len(tabla), '| conteos:', tabla['TOTAL'].sum())              # 78 y 8492

meses_con_datos = (tabla[MESES] > 0).sum(axis=1)
ESCENARIOS = {
    'E2': tabla[meses_con_datos == 12],
    'E3': tabla[tabla['TOTAL'] >= 30],
    'E5': tabla[(tabla['TOTAL'] >= 30) & (tabla['categoria'] != 'SISTEMA OSTEOARTICULAR')],
}
for e, s in ESCENARIOS.items():
    print(f'{e}: {len(s)} categorías, {s["TOTAL"].sum()} registros')          # 22/7830, 26/8048, 25/5538

# ---------------------------------------------------------------- representaciones
def representar(C, rep):
    if rep == 'proporciones':
        return C / C.sum(axis=1, keepdims=True)                              # ecuación (1)
    L = np.log((C + 0.5) / (C + 0.5).sum(axis=1, keepdims=True))
    return L - L.mean(axis=1, keepdims=True)                                 # CLR

def residuos_pearson(C):                                                     # ecuación (3)
    T = C.sum(axis=1); p = C.sum(axis=0) / C.sum()
    E = np.outer(T, p)
    return (C - E) / np.sqrt(E)

# ---------------------------------------------------------------- criterios
def sil(X, l):
    m = l != -1                                                              # DBSCAN: sin el ruido
    u = set(l[m])
    return silhouette_score(X[m], l[m]) if 2 <= len(u) < m.sum() else np.nan

def correr(alg, X, k=None, eps=None):
    if alg == 'KMeans': return KMeans(k, random_state=42, n_init=20).fit_predict(X)
    if alg == 'Ward':   return AgglomerativeClustering(k, linkage='ward').fit_predict(X)
    if alg == 'GMM':    return GaussianMixture(k, covariance_type='spherical', random_state=42, n_init=3).fit(X).predict(X)
    if alg == 'DBSCAN': return DBSCAN(eps=eps, min_samples=3).fit_predict(X)

def criterios(C, rep, detalle_dbscan=False):
    X = representar(C, rep)
    o = {}
    for a in ['KMeans', 'Ward', 'GMM']:
        o[f'sil_{a}'] = np.nanmax([sil(X, correr(a, X, k=k)) for k in KS])
    radios = np.percentile(pdist(X), [10, 20, 30, 40, 50])
    db = []
    for pct, eps in zip([10, 20, 30, 40, 50], radios):
        l = correr('DBSCAN', X, eps=eps)
        db.append({'percentil': pct, 'eps': eps, 'grupos': len(set(l) - {-1}),
                   'ruido': int((l == -1).sum()), 'silhouette': sil(X, l), 'etiquetas': l})
    v = [d['silhouette'] for d in db]
    o['sil_DBSCAN'] = np.nan if np.all(np.isnan(v)) else np.nanmax(v)
    # gap y R2 con K-Means: log W_k para k = 1..6
    o['logW_k1'] = np.log(((X - X.mean(0)) ** 2).sum())
    for k in KS:
        o[f'logW_k{k}'] = np.log(KMeans(k, random_state=42, n_init=20).fit(X).inertia_)
    o['R2'] = 1 - np.exp(o['logW_k2'] - o['logW_k1'])                         # ecuación (5), k = 2
    Z = residuos_pearson(C)
    o['Z'] = 1 - KMeans(2, random_state=42, n_init=10).fit(Z).inertia_ / ((Z - Z.mean(0)) ** 2).sum()
    return (o, db) if detalle_dbscan else o

def gap_k(obs, nulos):
    """Regla de Tibshirani et al. (2001) con el modelo nulo como referencia (igual que comun.gap_khat)."""
    m = {k: nulos[f'logW_k{k}'].mean() for k in range(1, 7)}
    s = {k: nulos[f'logW_k{k}'].std() * np.sqrt(1 + 1 / len(nulos)) for k in range(1, 7)}
    g = {k: m[k] - obs[f'logW_k{k}'] for k in range(1, 7)}
    for k in range(1, 6):
        if g[k] >= g[k + 1] - s[k + 1]:
            return k
    return 6

# ---------------------------------------------------------------- análisis
SALIDA.mkdir(exist_ok=True)
filas, filas_db = [], []
NOMBRES = {'sil_KMeans': 'Silhouette máximo, K-Means', 'sil_Ward': 'Silhouette máximo, Ward',
           'sil_GMM': 'Silhouette máximo, GMM', 'sil_DBSCAN': 'Silhouette máximo, DBSCAN',
           'R2': 'Reducción de dispersión R2', 'Z': 'R2 sobre residuos de Pearson (Z)'}
for ie, (e, s) in enumerate(ESCENARIOS.items()):
    C = s[MESES].to_numpy(float)
    T = C.sum(axis=1).astype(int); p = C.sum(axis=0) / C.sum()
    for ir, rep in enumerate(REPRESENTACIONES):
        t0 = time.time()
        obs, db = criterios(C, rep, detalle_dbscan=True)
        for d in db:
            filas_db.append({'escenario': e, 'representacion': rep, 'percentil_radio': d['percentil'],
                             'eps': round(d['eps'], 4), 'grupos': d['grupos'], 'categorias_ruido': d['ruido'],
                             'de': len(T), 'silhouette': d['silhouette'],
                             'nombres_ruido': ' | '.join(s['categoria'].to_numpy()[d['etiquetas'] == -1])})
        sims = []
        for b in range(B):
            rng = np.random.default_rng([SEMILLA, 500 + ie, ir, b])
            S = np.vstack([rng.multinomial(t, p) for t in T]).astype(float)       # modelo nulo N0
            sims.append(criterios(S, rep))
            if (b + 1) % 100 == 0:
                print(f'  {e} {rep}: {b + 1}/{B} simulaciones ({time.time() - t0:.0f} s)')
        sims = pd.DataFrame(sims)
        sims.to_csv(SALIDA / f'nulos_{e}_{rep}.csv', index=False)
        for c, nombre in NOMBRES.items():
            x = sims[c].dropna(); v = obs[c]
            filas.append({'escenario': e, 'representacion': rep, 'medida': nombre,
                          'valor': v, 'p_empirico': np.nan if pd.isna(v) else (1 + (x >= v).sum()) / (1 + len(x)),
                          'simulaciones_validas': len(x)})
        filas.append({'escenario': e, 'representacion': rep, 'medida': 'Gap (k elegido)',
                      'valor': gap_k(obs, sims), 'p_empirico': np.nan, 'simulaciones_validas': len(sims)})
        print(f'{e} {rep}: listo en {time.time() - t0:.0f} s')

res = pd.DataFrame(filas)
res.to_csv(SALIDA / 'tabla_E5_CLR.csv', index=False, sep=';', decimal=',', encoding='utf-8-sig')
pd.DataFrame(filas_db).to_csv(SALIDA / 'dbscan_ruido.csv', index=False, sep=';', decimal=',', encoding='utf-8-sig')

pd.set_option('display.width', 200)
print('\n=== Valor real y p empírico frente a N0 ===')
print(res.pivot_table(index='medida', columns=['escenario', 'representacion'],
                      values=['valor', 'p_empirico'], aggfunc='first').round(3).to_string())
print('\n=== DBSCAN en los datos reales (proporciones) ===')
print(pd.DataFrame(filas_db).query("representacion == 'proporciones'")
      [['escenario', 'percentil_radio', 'grupos', 'categorias_ruido', 'de']].to_string(index=False))
print('\nArchivos guardados en', SALIDA)
