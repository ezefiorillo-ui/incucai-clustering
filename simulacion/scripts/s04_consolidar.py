"""Consolidación: tablas finales y figuras del artículo. Solo lee resultados existentes (no simula)."""
import numpy as np, pandas as pd, json, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from comun import *
R_=os.path.join(RAIZ,'resultados'); F=os.path.join(RAIZ,'figuras'); os.makedirs(F,exist_ok=True)
cp=pd.read_csv(os.path.join(R_,'control_positivo.csv')); cp['kok']=(cp.k_elegido==cp.G)
# --- Tabla control positivo ---
g=cp.groupby(['escenario','G','delta','algoritmo'])
T=g.agg(separacion=('separacion','first'),ARI_elegido_media=('ARI_elegido','mean'),ARI_oraculo_media=('ARI_oraculo','mean'),
        recuperacion_ARI08=('ARI_elegido',lambda x:(x>=0.8).mean()),deteccion=('detecta','mean'),
        k_correcto=('kok','mean'),frac_ruido_media=('frac_ruido','mean')).reset_index()
T.round(3).to_csv(os.path.join(R_,'tabla_control_positivo.csv'),index=False)
# --- Dispersión verdadera estimada en datos reales (momentos): E||p̂_i-p||² = d_i² + Σp(1-p)/T_i ---
disp={}
for e in ['E2','E3']:
    s=escenario(e); C=s[MESES].values.astype(float); Ti=C.sum(1); p=C.sum(0)/C.sum(); Ph=C/Ti[:,None]
    obs2=((Ph-p)**2).sum(1); ruido2=(p*(1-p)).sum()/Ti
    exceso=np.maximum(obs2-ruido2,0); disp[e]={'rms_desvio_verdadero':float(np.sqrt((obs2-ruido2).mean().clip(0))),
       'rms_desvio_sin_osteo':float(np.sqrt((obs2-ruido2)[s.Codigo.values!=15].mean().clip(0))),
       'mediana_ruido':float(np.median(np.sqrt(ruido2)))}
json.dump(disp,open(os.path.join(R_,'dispersion_real_estimada.json'),'w'),indent=1)
print('Dispersión real estimada:',json.dumps(disp,indent=1))
# En el sintético, la distancia de cada categoría al perfil medio = separacion/2 (G=2) aprox.
# --- Tabla nulos ---
pv=pd.read_csv(os.path.join(R_,'pvalores_nulos.csv'))
tab=pv.pivot_table(index=['escenario','estadistico','observado'],columns='nulo',values='p_empirico').reset_index()
tab=tab[['escenario','estadistico','observado']+[c for c in ['N0','N1','N2','N1x2'] if c in tab.columns]]
tab.round(4).to_csv(os.path.join(R_,'tabla_nulos.csv'),index=False); print(tab.round(3).to_string(index=False))
# --- Figura 1: potencia/recuperación ---
col={'KMeans':'tab:blue','Ward':'tab:orange','GMM':'tab:green','DBSCAN':'tab:red'}
fig,ax=plt.subplots(2,4,figsize=(17,7.5),sharex=True,sharey=True)
for j,(e,Gk) in enumerate([('E2',2),('E2',3),('E3',2),('E3',3)]):
    d=T[(T.escenario==e)&(T.G==Gk)]
    for a in ALGOS:
        x=d[d.algoritmo==a]
        ax[0,j].plot(x.separacion,x.ARI_elegido_media,'o-',c=col[a],label=a)
        if a!='DBSCAN': ax[0,j].plot(x.separacion,x.ARI_oraculo_media,'--',c=col[a],alpha=.5)
        ax[1,j].plot(x.separacion,x.deteccion,'o-',c=col[a],label=a)
    for i in (0,1):
        ax[i,j].axvline(2*disp[e]['rms_desvio_verdadero'],c='k',ls=':',lw=1)
    ax[1,j].axhline(.05,c='gray',ls='--',lw=.8)
    ax[0,j].set_title(f'{e} (N={len(escenario(e))}), G={Gk}'); ax[1,j].set_xlabel('separación entre perfiles de grupo')
ax[0,0].set_ylabel('ARI medio vs etiquetas verdaderas\n(continuo: k elegido; trazos: k oráculo)'); ax[1,0].set_ylabel('tasa de detección vs nulo N0 (α=0,05)')
ax[0,0].legend(fontsize=8); fig.suptitle('Control positivo (R=100 por punto). Línea punteada vertical: 2× dispersión verdadera estimada en datos reales',fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(F,'fig_control_positivo.png'),dpi=150); plt.close()
# --- Figura 2: sensibilidad a sobredispersión (p por nulo) ---
fig,ax=plt.subplots(1,2,figsize=(12,4),sharey=True)
for j,e in enumerate(['E2','E3']):
    x=pv[(pv.escenario==e)&(pv.estadistico.isin(['sil_KMeans','sil_Ward','sil_GMM']))]
    for a in ['KMeans','Ward','GMM']:
        y=x[x.estadistico==f'sil_{a}'].set_index('nulo').reindex(['N0','N2','N1','N1x2'])
        ax[j].plot(['N0\nmultinomial','N2\nDM tejidos','N1\nDM global','N1×2\n(B=300)'],y.p_empirico,'o-',c=col[a],label=a)
    ax[j].axhline(.05,c='k',ls='--',lw=.8); ax[j].set_ylim(0,1); ax[j].set_title(f'{e}: p empírico del silhouette según modelo nulo')
ax[0].set_ylabel('p empírico'); ax[0].legend(); plt.tight_layout(); plt.savefig(os.path.join(F,'fig_sensibilidad_nulos.png'),dpi=150); plt.close()
# --- Figura 3: distribución nula Ward E3 por modelo ---
fig,ax=plt.subplots(figsize=(7,4))
for mnull,c in [('N0','gray'),('N2','tab:purple'),('N1','tab:green')]:
    ax.hist(pd.read_csv(os.path.join(R_,f'nulo_E3_{mnull}.csv')).sil_Ward,40,histtype='step',lw=1.5,color=c,label=mnull,density=True)
ax.axvline(0.3983,c='crimson',lw=2,label='observado 0,398'); ax.set_xlabel('silhouette (Ward, k elegido)'); ax.set_title('E3: distribución nula de Ward (B=1000)'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(F,'fig_nulo_ward_E3.png'),dpi=150); plt.close()
print(T[T.delta.isin([0,0.2,0.3,0.5])].pivot_table(index=['escenario','G','delta'],columns='algoritmo',values=['ARI_elegido_media','deteccion']).round(2).to_string())
