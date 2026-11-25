"""Experimento 4: estabilidad de particiones en datos reales (E2, E3). 200 perturbaciones por tipo.
 (a) multinomial: x_b ~ Mult(T_i, p̂_i)       (b) Dirichlet-multinomial: p_i ~ Dir(alpha_i p̂_i), alpha de N2 (tejidos)
 (c) submuestreo del 80 % de categorías sin reemplazo; ARI calculado SOLO sobre las categorías presentes (alineado)
 Semillas: K-Means y GMM con random_state 0..49 sobre los datos originales; ARI contra la solución random_state=42.
 Fascia lata: frecuencia con que ambas FASCIA LATA quedan juntas en un cluster de tamaño <=3."""
import numpy as np, pandas as pd, os, json
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.cluster.hierarchy import linkage, fcluster
from comun import RAIZ, SEED, MESES, escenario
from s06_deteccion import alphas_de
OUT=os.path.join(RAIZ,'resultados')
def fit(a,X,k,rs=42):
    if a=='KMeans': return KMeans(k,random_state=rs,n_init=20).fit_predict(X)
    if a=='Ward': return fcluster(linkage(X,'ward'),k,'maxclust')-1
    return GaussianMixture(k,covariance_type='spherical',random_state=rs,n_init=3).fit(X).predict(X)
rows=[]; fl=[]
for ie,e in enumerate(['E2','E3']):
    s=escenario(e); C=s[MESES].values.astype(float); T=C.sum(1).astype(int); P=C/T[:,None]; al=alphas_de(s,'N2',e)
    ifl=np.where(s.Nombre.str.contains('FASCIA LATA').values)[0]
    for a in ['KMeans','Ward','GMM']:
        for k in (2,3):
            L0=fit(a,P,k)
            for tipo in ['multinomial','dirichlet_mult','submuestreo_80']:
                rng=np.random.default_rng([SEED,300+ie,k,['multinomial','dirichlet_mult','submuestreo_80'].index(tipo)]); ar=[]; juntas=[]
                for b in range(200):
                    if tipo=='submuestreo_80':
                        idx=np.sort(rng.choice(len(T),int(0.8*len(T)),replace=False)); Lb=fit(a,P[idx],k); ar.append(ARI(L0[idx],Lb)); lab=np.full(len(T),-1); lab[idx]=Lb
                    else:
                        Cb=np.vstack([rng.multinomial(t,q if np.isinf(x) else rng.dirichlet(x*q)) for t,q,x in zip(T,P,al if tipo=='dirichlet_mult' else np.full(len(T),np.inf))]).astype(float)
                        lab=fit(a,Cb/Cb.sum(1,keepdims=True),k); ar.append(ARI(L0,lab))
                    if len(ifl)==2 and (lab[ifl]>=0).all(): juntas.append(lab[ifl[0]]==lab[ifl[1]] and (lab==lab[ifl[0]]).sum()<=3)
                rows.append({'escenario':e,'algoritmo':a,'k':k,'perturbacion':tipo,'ARI_media':np.mean(ar),'ARI_p10':np.percentile(ar,10),'ARI_p90':np.percentile(ar,90),
                             'tamaños_original':str(sorted(np.bincount(L0).tolist())),'fascia_juntas_aisladas':np.mean(juntas) if juntas else np.nan})
            if a!='Ward':
                sem=[ARI(L0,fit(a,P,k,rs)) for rs in range(50)]
                rows.append({'escenario':e,'algoritmo':a,'k':k,'perturbacion':'semillas_0_49','ARI_media':np.mean(sem),'ARI_p10':np.percentile(sem,10),'ARI_p90':np.percentile(sem,90),'tamaños_original':str(sorted(np.bincount(L0).tolist()))})
R=pd.DataFrame(rows); R.to_csv(os.path.join(OUT,'estabilidad_real.csv'),index=False); print(R.round(3).to_string(index=False))
