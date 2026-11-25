"""Completa, con verificación, el ARI de K-Means para el k elegido por el gap en réplicas ARCHIVADAS de det_eval.csv que
no guardaron ese k (versiones de s06 anteriores a v6 solo guardaban k=2 y 3). Regenera cada réplica con su semilla original
y el mismo código de s06, verifica log W (k=1..6) y ARI k=2,3 contra lo archivado (tolerancia 1e-12) y solo entonces guarda.
Con el s06 actual (que guarda todos los k) no hay nada que completar y el archivo queda vacío. Uso: python s06b_completar_ari_gap.py"""
import numpy as np, pandas as pd, os, sys
from comun import RAIZ, SEED, MESES, escenario, gap_khat
from s06_deteccion import alphas_de, generar, stats
R_=os.path.join(RAIZ,'resultados'); ev=pd.read_csv(os.path.join(R_,'det_eval.csv')); rows=[]; m=np.arange(12)
for (e,ru),sub in ev.groupby(['escenario','ruido']):
    nul=pd.read_csv(os.path.join(R_,f'det_null_{e}_{ru}.csv')); kh=gap_khat(sub,nul)
    s=escenario(e); T=s.TOTAL.values.astype(int); p0=s[MESES].values.sum(0)/T.sum(); al=alphas_de(s,ru,e); ie,im=['E2','E3'].index(e),['N0','N2','N2x2'].index(ru)
    for (ix,r),k in zip(sub.iterrows(),kh):
        if k==1 or (f'ARI_km_k{k}' in sub.columns and pd.notna(r.get(f'ARI_km_k{k}'))): continue
        G,d,rep=int(r.G),float(r.delta),int(r.replica)
        Pg=np.array([p0*(1+d*np.sin(2*np.pi*m/12+2*np.pi*g/G)) for g in range(G)]); Pg/=Pg.sum(1,keepdims=True)
        rng=np.random.default_rng([SEED,200+ie,im,G,int(round(d*100)),rep]); lab=rng.permutation(np.arange(len(T))%G)
        o=stats(generar(T,Pg[lab],al,rng),truth=lab,G=G,rng=rng)
        el=max(abs(o[f'logWkm_k{j}']-r[f'logWkm_k{j}']) for j in range(1,7)); ea=max(abs(o[f'ARI_km_k{j}']-r[f'ARI_km_k{j}']) for j in (2,3))
        if el>1e-12 or ea>1e-12: sys.exit(f'ERROR: la regeneración de {e}/{ru}/G{G}/δ{d}/réplica {rep} no reproduce lo archivado (logW {el:.2e}, ARI {ea:.2e})')
        rows.append({'escenario':e,'ruido':ru,'G':G,'delta':d,'replica':rep,'k_gap':int(k),'ARI_gap':o[f'ARI_km_k{k}'],'max_error_logW':el,'max_error_ARI_k2_k3':ea,
                     'semilla':f'[{SEED},{200+ie},{im},{G},{int(round(d*100))},{rep}]'})
pd.DataFrame(rows,columns=['escenario','ruido','G','delta','replica','k_gap','ARI_gap','max_error_logW','max_error_ARI_k2_k3','semilla']).to_csv(os.path.join(R_,'gap_ari_complemento.csv'),index=False)
print('réplicas completadas y verificadas:',len(rows))
