"""Experimentos 1-3: diagnóstico de detección y estadísticos alternativos, calculados en UNA pasada por dataset.
Uso:
  python s06_deteccion.py null <E2|E3> <N0|N2|N2x2> <B>          -> resultados/det_null_<E>_<mod>.csv
  python s06_deteccion.py eval <E2|E3> <N0|N2|N2x2> <R> <Gs> <deltas>  -> resultados/det_eval.csv
Guardado progresivo cada 50 datasets; reanuda si ya existen filas.
Estadísticos (todos en el espacio de perfiles P=x/T, euclídeo, salvo *_Z):
  A   sil_max_km / sil_max_w : máximo silhouette sobre k=2..6 (K-Means / Ward). Grande => estructura.
  B   sil en k=G (oráculo, solo diagnóstico).
  C   sil_km_k2, sil_km_k3, sil_w_k2, sil_w_k3 : silhouette con k prefijado.
  R   Rkm_k = 1 - W_k/W_1 : reducción de dispersión intra-cluster (K-Means), k prefijado. Grande => estructura.
  gap logWkm_k (k=1..6): se combina en s07 con la referencia nula para el gap statistic (Tibshirani 2001)
      con referencia = modelo nulo (no uniforme).
  stab_w_k : estabilidad Ward k prefijado = ARI medio entre partición original y 15 re-muestreos multinomiales
      paramétricos de cada categoría (mismas observaciones => ARI alineado). Grande => estructura.
  *_Z: K-Means sobre residuos de Pearson z=(x-T p̂)/sqrt(T p̂) (p̂ pooled del propio dataset), que homogeneizan la
      varianza de ruido entre categorías de distinto volumen (diagnóstico del porqué).
"""
import numpy as np, pandas as pd, sys, os, json, time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score as ARI
from scipy.cluster.hierarchy import linkage, fcluster
from comun import RAIZ, SEED, MESES, escenario
OUT=os.path.join(RAIZ,'resultados'); SD=json.load(open(os.path.join(OUT,'sobredispersion.json')))
KS=range(2,7); TEJ=['Osteo-tendinoso especifico','Agregado osteoarticular']
def alphas_de(s,mod,e):
    tej=s.Grupo.isin(TEJ).values; r=SD[e]['rho_tejidos']*(2 if mod=='N2x2' else 1)
    return np.full(len(s),np.inf) if mod=='N0' else np.where(tej,1/r-1,np.inf)
def generar(T,prof_por_obs,alph,rng):
    return np.vstack([rng.multinomial(int(t),q) if np.isinf(a) else rng.multinomial(int(t),rng.dirichlet(a*q))
                      for t,q,a in zip(T,prof_por_obs,alph)]).astype(float)
def sil(X,l): return silhouette_score(X,l) if 1<len(set(l))<len(l) else np.nan
def ward_labels(X,ks=range(1,7)):
    L=linkage(X,'ward'); return {k:fcluster(L,k,'maxclust')-1 for k in ks}
def W(X,l): return sum(((X[l==c]-X[l==c].mean(0))**2).sum() for c in set(l))
def stats(C,truth=None,G=None,rng=None):
    T=C.sum(1); X=C/T[:,None]; o={}; lab={}
    for k in KS:
        km=KMeans(k,random_state=42,n_init=20).fit(X); lab[('km',k)]=km.labels_
        o[f'sil_km_k{k}']=sil(X,km.labels_); o[f'logWkm_k{k}']=np.log(km.inertia_)
    o['logWkm_k1']=np.log(((X-X.mean(0))**2).sum())
    wl=ward_labels(X)
    for k in KS: lab[('w',k)]=wl[k]; o[f'sil_w_k{k}']=sil(X,wl[k])
    for pre,t in [('km','sil_km_k'),('w','sil_w_k')]:
        v=[o[f'{t}{k}'] for k in KS]; kb=KS[int(np.nanargmax(v))]
        o[f'sil_max_{pre}']=np.nanmax(v); o[f'kA_{pre}']=kb; o[f'minsize_A_{pre}']=int(np.bincount(lab[(pre,kb)]).min())
    W1=np.exp(o['logWkm_k1'])
    for k in (2,3): o[f'Rkm_k{k}']=1-np.exp(o[f'logWkm_k{k}'])/W1
    # residuos de Pearson
    ph=C.sum(0)/C.sum(); Z=(C-np.outer(T,ph))/np.sqrt(np.outer(T,ph)); WZ1=((Z-Z.mean(0))**2).sum()
    for k in (2,3):
        km=KMeans(k,random_state=42,n_init=10).fit(Z); lab[('z',k)]=km.labels_
        o[f'sil_Z_k{k}']=sil(Z,km.labels_); o[f'R_Z_k{k}']=1-km.inertia_/WZ1
    # estabilidad Ward (bootstrap paramétrico multinomial, mismas observaciones)
    rs=rng if rng is not None else np.random.default_rng(0)
    for k in (2,3):
        a=[ARI(lab[('w',k)],ward_labels((Cb:=np.vstack([rs.multinomial(int(t),q) for t,q in zip(T,X)]).astype(float))/Cb.sum(1,keepdims=True),[k])[k]) for _ in range(15)]
        o[f'stab_w_k{k}']=float(np.mean(a))
    if truth is not None:
        o['ARI_A_km']=ARI(truth,lab[('km',o['kA_km'])]); o['ARI_A_w']=ARI(truth,lab[('w',o['kA_w'])])
        for k in (2,3):
            o[f'ARI_km_k{k}']=ARI(truth,lab[('km',k)]); o[f'ARI_w_k{k}']=ARI(truth,lab[('w',k)]); o[f'ARI_Z_k{k}']=ARI(truth,lab[('z',k)])
        o['ARI_oraculo_km']=ARI(truth,lab[('km',G)]); o['sil_oraculo_km']=o[f'sil_km_k{G}']
        for k in KS: o[f'ARI_km_k{k}']=ARI(truth,lab[('km',k)])  # todos los k: necesario para el ARI del gap (k̂ puede ser 4–6)
    return o
def guardar(rows,f): pd.DataFrame(rows).to_csv(f,mode='a',header=not os.path.exists(f),index=False)
if __name__=='__main__':
    modo,e,mod=sys.argv[1],sys.argv[2],sys.argv[3]; ie,im=['E2','E3'].index(e),['N0','N2','N2x2'].index(mod)
    s=escenario(e); T=s.TOTAL.values.astype(int); p0=s[MESES].values.sum(0)/T.sum(); al=alphas_de(s,mod,e); t0=time.time()
    if modo=='null':
        B=int(sys.argv[4]); f=os.path.join(OUT,f'det_null_{e}_{mod}.csv'); hecho=len(pd.read_csv(f)) if os.path.exists(f) else 0; rows=[]
        for b in range(hecho,B):
            rng=np.random.default_rng([SEED,100+ie,im,b]); C=generar(T,[p0]*len(T),al,rng)
            r=stats(C,rng=rng); r['b']=b; rows.append(r)
            if len(rows)==50: guardar(rows,f); rows=[]
        if rows: guardar(rows,f)
    else:
        R=int(sys.argv[4]); Gs=[int(x) for x in sys.argv[5].split(',')]; Ds=[float(x) for x in sys.argv[6].split(',')]
        f=os.path.join(OUT,'det_eval.csv'); old=pd.read_csv(f) if os.path.exists(f) else None; m=np.arange(12)
        for G in Gs:
            for d in Ds:
                if old is not None and ((old.escenario==e)&(old.ruido==mod)&(old.G==G)&np.isclose(old.delta,d)).sum()>=R: continue
                Pg=np.array([p0*(1+d*np.sin(2*np.pi*m/12+2*np.pi*g/G)) for g in range(G)]); Pg/=Pg.sum(1,keepdims=True); rows=[]
                for r in range(R):
                    rng=np.random.default_rng([SEED,200+ie,im,G,int(round(d*100)),r]); lab=rng.permutation(np.arange(len(T))%G)
                    C=generar(T,Pg[lab],al,rng); o=stats(C,truth=lab,G=G,rng=rng)
                    o.update(escenario=e,ruido=mod,G=G,delta=d,replica=r); rows.append(o)
                guardar(rows,f)
    print(modo,e,mod,'ok',round(time.time()-t0),'s')
