import numpy as np, pandas as pd, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from comun import RAIZ
R_=os.path.join(RAIZ,'resultados'); F=os.path.join(RAIZ,'figuras'); os.makedirs(F,exist_ok=True)
T=pd.read_csv(os.path.join(R_,'tabla_potencia.csv')); RE=pd.read_csv(os.path.join(R_,'reales_todos_estadisticos.csv')); D=pd.read_csv(os.path.join(R_,'diagnostico_maxsil.csv'))
SEL={'A  máx. silhouette K-Means (k=2–6)':('A máx. silhouette','tab:red','-'),'B  silhouette k verdadero (solo diagnóstico)':('B silhouette k verdadero','tab:red',':'),
 'R  reducción de dispersión K-Means k=2':('R reducción dispersión k=2','tab:blue','-'),'G  gap statistic (referencia = nulo)':('Gap (ref. nulo)','tab:green','-'),
 'S  estabilidad Ward k=2':('S estabilidad Ward k=2','tab:purple','-'),'Z  reducción dispersión residuos k=2':('Z residuos Pearson k=2','tab:gray','--')}
fig,ax=plt.subplots(2,2,figsize=(13,8.5),sharey=True)
for i,e in enumerate(['E2','E3']):
    for j,G in enumerate([2,3]):
        a=ax[i,j]; x=T[(T.escenario==e)&(T.ruido=='N0')&(T.G==G)]
        for pr,(lb,c,ls) in SEL.items():
            y=x[x.procedimiento==pr].sort_values('delta'); a.plot(y.delta,y.deteccion,ls,c=c,marker='o',ms=4,label=lb)
            a.fill_between(y.delta,y.IC95_inf,y.IC95_sup,color=c,alpha=.08)
        y=x[x.procedimiento=='B  silhouette k verdadero (solo diagnóstico)'].sort_values('delta'); a.plot(y.delta,y.ARI_medio,c='k',lw=2.5,alpha=.6,label='ARI K-Means con k verdadero (recuperación)')
        a.axhline(.05,c='k',ls='--',lw=.7); a.set_title(f'{e}, G={G} (ruido multinomial, R=100)'); a.set_xlabel('δ (intensidad de señal)')
ax[0,0].set_ylabel('tasa de detección (IC95 Wilson) / ARI'); ax[1,0].set_ylabel('tasa de detección (IC95 Wilson) / ARI'); ax[0,0].legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(os.path.join(F,'fig_recuperacion_vs_deteccion.png'),dpi=150); plt.close()
fig,ax=plt.subplots(1,2,figsize=(12,4))
for j,e in enumerate(['E2','E3']):
    d=D[D.escenario==e]; ax[j].bar(range(len(d)),d['frac_minsize<=2'],color=['gray']+['tab:blue']*3+['tab:orange']*3)
    ax[j].set_xticks(range(len(d)),d.datos,rotation=35,fontsize=8); ax[j].set_title(f'{e}: soluciones de máx. silhouette con algún cluster de ≤2 categorías')
plt.tight_layout(); plt.savefig(os.path.join(F,'fig_diagnostico_silhouette.png'),dpi=150); plt.close()
fig,ax=plt.subplots(2,4,figsize=(17,7),sharey=True)
for i,e in enumerate(['E2','E3']):
    for j,pr in enumerate(['A  máx. silhouette K-Means (k=2–6)','R  reducción de dispersión K-Means k=2','G  gap statistic (referencia = nulo)','Z  reducción dispersión residuos k=2']):
        for ru,c in [('N0','k'),('N2','tab:orange'),('N2x2','tab:red')]:
            y=T[(T.escenario==e)&(T.G==2)&(T.ruido==ru)&(T.procedimiento==pr)].sort_values('delta')
            ax[i,j].errorbar(y.delta,y.deteccion,yerr=[y.deteccion-y.IC95_inf,y.IC95_sup-y.deteccion],fmt='o-',c=c,capsize=2,label=ru)
        ax[i,j].axhline(.05,c='gray',ls='--',lw=.7); ax[i,j].set_title(f'{e}: {SEL[pr][0]}',fontsize=9)
ax[0,0].legend(); ax[0,0].set_ylabel('detección (G=2)'); ax[1,0].set_ylabel('detección (G=2)')
plt.tight_layout(); plt.savefig(os.path.join(F,'fig_potencia_sobredispersion.png'),dpi=150); plt.close()
pd.set_option('display.width',250)
print('== Datos reales: p según nulo ==')
print(RE.pivot_table(index=['escenario','procedimiento'],columns='ruido',values='p_empirico').round(3).to_string())
print('== FP (δ=0) y potencia G=2 por ruido ==')
x=T[(T.G==2)&(T.procedimiento.isin(SEL))]
print(x.pivot_table(index=['escenario','procedimiento'],columns=['ruido','delta'],values='deteccion').round(2).to_string())
print('== ARI por procedimiento, N0 ==')
print(T[(T.ruido=='N0')&T.delta.isin([0.3,0.5])].pivot_table(index='procedimiento',columns=['escenario','G','delta'],values='ARI_medio').round(2).to_string())
