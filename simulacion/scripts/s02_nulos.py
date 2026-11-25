"""Distribuciones nulas del pipeline de 4 algoritmos.
Uso: python s02_nulos.py <E2|E3|E5> <N0|N1|N2|N1x2> <B>
 N0  multinomial (sin sobredispersión)
 N1  Dirichlet-multinomial, rho global estimado por momentos (s01)
 N2  Dirichlet-multinomial, rho específico: tejidos osteo-tendinosos+agregado = rho_tejidos; resto rho=0
 N1x2 sensibilidad: rho global duplicado (cota superior ante el sesgo negativo del estimador)
El perfil medio p es el pooled del escenario (MLE bajo H0). Mismo pipeline y grilla que en datos reales."""
import numpy as np, pandas as pd, json, sys, os, time
from comun import *
e,mod,B=sys.argv[1],sys.argv[2],int(sys.argv[3]); OUT=os.path.join(RAIZ,'resultados')
sd=json.load(open(os.path.join(OUT,'sobredispersion.json')))[e if e!='E5' else 'E3']
s=escenario(e); C=s[MESES].values.astype(float); T=C.sum(1); p=C.sum(0)/C.sum()
tej=s.Grupo.isin(['Osteo-tendinoso especifico','Agregado osteoarticular']).values
inv=lambda r: np.inf if r<=0 else 1/r-1
if mod=='N0': alphas=np.full(len(T),np.inf)
elif mod=='N1': alphas=np.full(len(T),inv(sd['rho_global']))
elif mod=='N1x2': alphas=np.full(len(T),inv(2*sd['rho_global']))
elif mod=='N2': alphas=np.where(tej,inv(sd['rho_tejidos']),np.inf)
def simular(rng):
    return np.vstack([rng.multinomial(int(t),p) if np.isinf(a) else rng.multinomial(int(t),rng.dirichlet(a*p)) for t,a in zip(T,alphas)]).astype(float)
def chi2(C):
    q=C.sum(0)/C.sum(); E=np.outer(C.sum(1),q); return ((C-E)**2/E).sum()
obs,_=pipeline(perfiles(C)); fo={f'sil_{a}':obs[a][0] for a in ALGOS}; fo['chi2']=chi2(C)
rng=np.random.default_rng([SEED,['E2','E3','E5'].index(e),['N0','N1','N2','N1x2'].index(mod)]) ; t=time.time(); rows=[]
for b in range(B):
    S=simular(rng); o,_=pipeline(perfiles(S)); r={f'sil_{a}':o[a][0] for a in ALGOS}; r['chi2']=chi2(S); rows.append(r)
sim=pd.DataFrame(rows); sim.to_csv(os.path.join(OUT,f'nulo_{e}_{mod}.csv'),index=False)
res=[]
for c in fo:
    x=sim[c].dropna(); o=fo[c]
    res.append({'escenario':e,'nulo':mod,'B':B,'estadistico':c,'observado':o,'nulo_media':x.mean(),'nulo_p95':x.quantile(.95),
                'p_empirico':np.nan if pd.isna(o) else (1+(x>=o).sum())/(1+len(x)),'sims_validas':len(x)})
R=pd.DataFrame(res); f=os.path.join(OUT,'pvalores_nulos.csv')
old=pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
pd.concat([old[~((old.get('escenario')==e)&(old.get('nulo')==mod))] if len(old) else old,R]).to_csv(f,index=False)
print(R.round(4).to_string(index=False)); print('seg',round(time.time()-t))
