"""Control positivo: datos sintéticos con grupos conocidos y los totales anuales reales del escenario.
Uso: python s03_control_positivo.py <E2|E3> <G> [R]
Diseño:
 - p0 = perfil mensual pooled del escenario real.
 - Perfil del grupo g: p_g ∝ p0 * (1 + δ·sen(2πm/12 + 2πg/G)), m=0..11.
 - Etiquetas balanceadas (i mod G) permutadas al azar en cada réplica, independientes del volumen T_i.
 - Conteos: x_i ~ Multinomial(T_i, p_{g(i)}).  δ=0 => sin grupos (evalúa falsos positivos).
 - Mismo pipeline y grilla que los datos reales (comun.pipeline). Además k verdadero (oráculo) para KMeans/Ward/GMM.
 - Detección: silhouette > percentil 95 del nulo multinomial N0 del mismo escenario y algoritmo (B=1000, s02).
 - DBSCAN: los puntos de ruido (-1) se tratan como UNA etiqueta adicional al calcular el ARI (penaliza dejar
   observaciones sin asignar); el silhouette excluye el ruido; se reporta la fracción de ruido aparte.
Semillas: default_rng([SEED, escenario, G, índice δ, réplica]). Guardado progresivo por δ."""
import numpy as np, pandas as pd, sys, os, time
from sklearn.metrics import adjusted_rand_score
from comun import *
e=sys.argv[1]; G=int(sys.argv[2]); R=int(sys.argv[3]) if len(sys.argv)>3 else 100
DELTAS=[0,0.1,0.2,0.3,0.5,0.8]; OUT=os.path.join(RAIZ,'resultados'); f=os.path.join(OUT,'control_positivo.csv')
s=escenario(e); T=s.TOTAL.values.astype(int); N=len(T); p0=s[MESES].values.sum(0)/T.sum()
p95=pd.read_csv(os.path.join(OUT,f'nulo_{e}_N0.csv')).quantile(.95)
m=np.arange(12); t0=time.time()
for di,d in enumerate(DELTAS):
    if os.path.exists(f):
        old=pd.read_csv(f)
        if ((old.escenario==e)&(old.G==G)&(old.delta==d)).sum()>=R*4: continue
    P=np.array([p0*(1+d*np.sin(2*np.pi*m/12+2*np.pi*g/G)) for g in range(G)]); P/=P.sum(1,keepdims=True)
    sep=np.mean([np.linalg.norm(P[a]-P[b]) for a in range(G) for b in range(a+1,G)])
    rows=[]
    for r in range(R):
        rng=np.random.default_rng([SEED,['E2','E3'].index(e),G,di,r])
        lab=rng.permutation(np.arange(N)%G)
        C=np.vstack([rng.multinomial(t,P[l]) for t,l in zip(T,lab)]).astype(float)
        out,orc=pipeline(perfiles(C),oracle_k=G)
        for a in ALGOS:
            s_,l,par=out[a]
            rows.append({'escenario':e,'N':N,'G':G,'delta':d,'separacion':sep,'replica':r,'algoritmo':a,
              'silhouette':s_,'param':par,'k_elegido':len(set(l)-{-1}),
              'ARI_elegido':adjusted_rand_score(lab,l),
              'ARI_oraculo':adjusted_rand_score(lab,orc[a]) if a in orc else np.nan,
              'detecta':bool(not np.isnan(s_) and s_>p95[f'sil_{a}']),
              'frac_ruido':float((l==-1).mean())})
    pd.DataFrame(rows).to_csv(f,mode='a',header=not os.path.exists(f),index=False)
    print(e,G,'δ=',d,'sep=%.3f'%sep,'ok',round(time.time()-t0),'s',flush=True)
