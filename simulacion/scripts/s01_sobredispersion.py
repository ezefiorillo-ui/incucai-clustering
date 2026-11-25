"""Estimación de sobredispersión (Dirichlet-multinomial) por método de momentos y chequeo de ajuste.
H0 DM: todas las categorías comparten el perfil medio p; cada categoría i tiene pi_i ~ Dir(alpha*p),
x_i ~ Mult(T_i, pi_i). rho = 1/(1+alpha) es la correlación intra-clase; inflación de varianza = 1+(T_i-1)rho.
Estimador MoM: E[chi2_i] ~ (K-1)(1+(T_i-1)rho)  =>  rho = (chi2 - (K-1)(N-1)) / ((K-1) * sum(T_i-1)), truncado en 0.
"""
import numpy as np, pandas as pd, json, os
from comun import *
K=12; OUT=os.path.join(RAIZ,'resultados')
def chi2_tabla(C):
    p=C.sum(0)/C.sum(); E=np.outer(C.sum(1),p); return ((C-E)**2/E).sum(), ((C-E)**2/E).sum(1)
def rho_mom(C):
    chi,_=chi2_tabla(C); T=C.sum(1); N=len(T)
    return max(0.0,(chi-(K-1)*(N-1))/((K-1)*(T-1).sum())), chi
TEJ=['Osteo-tendinoso especifico','Agregado osteoarticular']
res={}; rng=np.random.default_rng(SEED)
for e in ['E2','E3']:
    s=escenario(e); C=s[MESES].values.astype(float); T=C.sum(1); p=C.sum(0)/C.sum()
    r,chi=rho_mom(C); tej=s.Grupo.isin(TEJ).values
    r_t,_=rho_mom(C[tej]); r_o,_=rho_mom(C[~tej])
    # Validación del estimador: simular DM con rho conocido y re-estimar
    val=[]
    for rr in [0.0,r,2*r]:
        a=np.inf if rr==0 else 1/rr-1
        est=[rho_mom(sim_dirmult(T,p,a,rng))[0] for _ in range(300)]
        val.append({'rho_verdadero':rr,'rho_est_media':float(np.mean(est)),'rho_est_sd':float(np.std(est))})
    # Chequeo de ajuste: chi2_i observado vs distribución bajo DM(rho global), por categoría
    a=1/r-1 if r>0 else np.inf
    sims=np.array([chi2_tabla(sim_dirmult(T,p,a,rng))[1] for _ in range(1000)])
    _,ci=chi2_tabla(C)
    pct=(sims<ci).mean(0)  # percentil del observado en su distribución DM
    fit=pd.DataFrame({'Categoria':s.Nombre,'Grupo':s.Grupo,'T':T.astype(int),'chi2_i_obs':ci.round(1),
        'chi2_i_DM_mediana':np.median(sims,0).round(1),'percentil_obs_bajo_DM':pct.round(3)})
    fit.to_csv(os.path.join(OUT,f'ajuste_DM_{e}.csv'),index=False)
    res[e]={'N':len(s),'chi2':chi,'rho_global':r,'alpha_global':(1/r-1) if r>0 else None,
            'rho_tejidos':r_t,'rho_no_tejidos':r_o,'n_tejidos':int(tej.sum()),
            'validacion_estimador':val,
            'frac_cat_debajo_p05_DM':float((pct<0.05).mean()),'frac_cat_encima_p95_DM':float((pct>0.95).mean())}
    print(e,json.dumps({k:v for k,v in res[e].items() if k!='validacion_estimador'},default=float))
    print(pd.DataFrame(val).round(4).to_string(index=False))
    print(fit.sort_values('T',ascending=False).head(8).to_string(index=False))
json.dump(res,open(os.path.join(OUT,'sobredispersion.json'),'w'),indent=1,default=float)
