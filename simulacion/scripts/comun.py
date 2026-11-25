"""Módulo común: datos, escenarios, pipeline de 4 algoritmos y generadores nulos.
Todas las funciones son deterministas dadas sus semillas."""
import numpy as np, pandas as pd, warnings, os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
warnings.filterwarnings("ignore")
RAIZ=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED=20261017
MESES=['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SET','OCT','NOV','DIC']
KS=range(2,7)
ALGOS=['KMeans','Ward','GMM','DBSCAN']

def cargar():
    return pd.read_csv(os.path.join(RAIZ,'datos','categorias_78_clasificadas.csv'),sep=';')
def escenario(nombre,d=None):
    d=cargar() if d is None else d
    if nombre=='E2': s=d[d.Meses==12]
    elif nombre=='E3': s=d[d.TOTAL>=30]
    elif nombre=='E5': s=d[(d.TOTAL>=30)&(d.Codigo!=15)]
    else: raise ValueError(nombre)
    return s.reset_index(drop=True)

def _valido(l): u=set(l)-{-1}; return len(u)>=2 and len(u)<len(l)
def sil(X,l):
    """Silhouette euclídeo. DBSCAN: puntos de ruido (-1) excluidos del cálculo.
    Puntos en clusters unitarios reciben s=0 (convención sklearn/Rousseeuw)."""
    m=l!=-1
    return silhouette_score(X[m],l[m]) if _valido(l[m]) else np.nan

def correr(alg,X,k=None,eps=None):
    if alg=='KMeans': return KMeans(k,random_state=42,n_init=20).fit_predict(X)
    if alg=='Ward': return AgglomerativeClustering(k,linkage='ward').fit_predict(X)
    if alg=='GMM': return GaussianMixture(k,covariance_type='spherical',random_state=42,n_init=3).fit(X).predict(X)
    if alg=='DBSCAN': return DBSCAN(eps=eps,min_samples=3).fit_predict(X)

def pipeline(X,oracle_k=None):
    """Selección interna por máximo silhouette con grilla idéntica en datos reales y simulados.
    KMeans/Ward/GMM: k=2..6. DBSCAN: min_samples=3, eps en percentiles 10..50 de distancias.
    Devuelve {alg:(silhouette, etiquetas, parametro)} y, si oracle_k, etiquetas con k verdadero."""
    grid={a:[(k,correr(a,X,k=k)) for k in KS] for a in ['KMeans','Ward','GMM']}
    grid['DBSCAN']=[(round(e,4),correr('DBSCAN',X,eps=e)) for e in np.percentile(pdist(X),[10,20,30,40,50])]
    out={}
    for a,cands in grid.items():
        sc=[(sil(X,l),l,p) for p,l in cands]
        out[a]=max(sc,key=lambda t:-1 if np.isnan(t[0]) else t[0])
    orc={a:correr(a,X,k=oracle_k) for a in ['KMeans','Ward','GMM']} if oracle_k else None
    return out,orc

def perfiles(C): return C/C.sum(1,keepdims=True)
def sim_multinomial(T,p,rng):
    return np.vstack([rng.multinomial(int(t),p) for t in T]).astype(float)
def sim_dirmult(T,p,alpha,rng):
    """Dirichlet-multinomial: pi_i ~ Dir(alpha*p); x_i ~ Mult(T_i, pi_i). alpha=inf equivale a multinomial."""
    if np.isinf(alpha): return sim_multinomial(T,p,rng)
    return np.vstack([rng.multinomial(int(t),rng.dirichlet(alpha*p)) for t in T]).astype(float)


def gap_khat(df,ref):
    """Regla del gap (Tibshirani et al., 2001) con referencia = conjunto nulo: Gap(k)=media_ref(logW_k)-logW_k,
    s_k=sd_ref*sqrt(1+1/B); k̂ = primer k en 1..5 con Gap(k) >= Gap(k+1)-s_(k+1); si ninguno, 6.
    Es una regla de selección: no es un contraste calibrado al 5 %."""
    m={k:ref[f'logWkm_k{k}'].mean() for k in range(1,7)}; s={k:ref[f'logWkm_k{k}'].std()*np.sqrt(1+1/len(ref)) for k in range(1,7)}
    g=np.column_stack([m[k]-df[f'logWkm_k{k}'] for k in range(1,7)]); out=[]
    for row in g:
        kh=6
        for k in range(1,6):
            if row[k-1]>=row[k]-s[k+1]: kh=k; break
        out.append(kh)
    return np.array(out)
