"""Consolida experimentos 1-3. Umbral de cada estadístico = percentil 95 de su nulo (mismo escenario y modelo de ruido,
semillas independientes de la evaluación). Falsos positivos = detección en réplicas delta=0. IC de Wilson 95 %."""
import numpy as np, pandas as pd, os, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from comun import RAIZ, MESES, escenario
from s06_deteccion import stats
R_=os.path.join(RAIZ,'resultados'); F=os.path.join(RAIZ,'figuras'); ev=pd.read_csv(os.path.join(R_,'det_eval.csv'))
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; c=(p+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); return (max(0,c-h),min(1,c+h))
PROC={'A  máx. silhouette K-Means (k=2–6)':('sil_max_km','ARI_A_km'),'A  máx. silhouette Ward (k=2–6)':('sil_max_w','ARI_A_w'),
 'B  silhouette k verdadero (solo diagnóstico)':('ORACULO','ARI_oraculo_km'),
 'C  silhouette K-Means k=2 prefijado':('sil_km_k2','ARI_km_k2'),'C  silhouette K-Means k=3 prefijado':('sil_km_k3','ARI_km_k3'),
 'C  silhouette Ward k=2 prefijado':('sil_w_k2','ARI_w_k2'),
 'R  reducción de dispersión K-Means k=2':('Rkm_k2','ARI_km_k2'),'R  reducción de dispersión K-Means k=3':('Rkm_k3','ARI_km_k3'),
 'Z  silhouette residuos Pearson k=2':('sil_Z_k2','ARI_Z_k2'),'Z  reducción dispersión residuos k=2':('R_Z_k2','ARI_Z_k2'),
 'S  estabilidad Ward k=2':('stab_w_k2','ARI_w_k2'),'S  estabilidad Ward k=3':('stab_w_k3','ARI_w_k3'),
 'G  gap statistic (referencia = nulo)':('GAP',None)}
from comun import gap_khat
import sys
CLAVE=['escenario','ruido','G','delta','replica']
COMP=os.path.join(R_,'gap_ari_complemento.csv')
# ARI_gap se lee como texto (sin convertir vacíos en NaN) para poder rechazar explícitamente vacíos y valores no numéricos.
comp=pd.read_csv(COMP,dtype={'ARI_gap':str},keep_default_na=False) if os.path.exists(COMP) else pd.DataFrame(columns=CLAVE+['k_gap','ARI_gap'])
if len(comp):
    dup=comp[comp.duplicated(CLAVE,keep=False)]
    if len(dup): sys.exit('ERROR: claves duplicadas en gap_ari_complemento.csv: '+'; '.join(f"{r.escenario}/{r.ruido}/G{r.G}/δ{r.delta}/réplica {r.replica}" for r in dup.itertuples()))
# --- Validación previa de TODAS las columnas ARI del archivo principal (det_eval.csv), antes de cualquier cálculo ---
# ARI_km_k4..k6 pueden faltar (NaN) en réplicas archivadas: se completan desde el complemento verificado de s06b.
OPC={'ARI_km_k4','ARI_km_k5','ARI_km_k6'}
for col in [x for x in ev.columns if x.startswith('ARI_')]:
    raw=ev[col]; num=pd.to_numeric(raw,errors='coerce')
    motivo=pd.Series('',index=ev.index)
    motivo[num.isna()&raw.notna()]='no numérico'; motivo[np.isinf(num)]='no finito (infinito)'
    motivo[np.isfinite(num)&(num.abs()>1)]='fuera de [-1, 1]'
    if col not in OPC: motivo[raw.isna()]='vacío o NaN'
    if (motivo!='').any():
        i=motivo[motivo!=''].index[0]; r=ev.loc[i]
        sys.exit(f"ERROR: ARI inválido ({motivo[i]}) en det_eval.csv, columna {col}, para {r.escenario}/{r.ruido}/G{r.G}/δ{r.delta}/réplica {r.replica}; valor={raw[i]!r}; {int((motivo!='').sum())} valor(es) inválido(s) en esa columna")
    ev[col]=num
def ari_valido(x,ctx):
    """Devuelve el ARI como float o detiene la ejecución si es vacío, no numérico, no finito o está fuera de [-1, 1]."""
    if x is None or (isinstance(x,str) and x.strip()==''): motivo='vacío'
    else:
        try: v=float(x)
        except (TypeError,ValueError): v=None; motivo='no numérico'
        if v is not None:
            if not np.isfinite(v): motivo='no finito (NaN o infinito)'
            elif not -1.0<=v<=1.0: motivo='fuera de [-1, 1]'
            else: return v
    sys.exit(f'ERROR: ARI del gap inválido ({motivo}) para {ctx}; valor={x!r}')
traza=[]
filas=[]
for (e,ruido),sub in ev.groupby(['escenario','ruido']):
    nul=pd.read_csv(os.path.join(R_,f'det_null_{e}_{ruido}.csv'))
    for nombre,(st,arc) in PROC.items():
        for (G,d),c in sub.groupby(['G','delta']):
            if st=='ORACULO': det=c.sil_oraculo_km.values>nul[f'sil_km_k{G}'].quantile(.95)
            elif st=='GAP': det=gap_khat(c,nul)>=2
            else: det=c[st].values>nul[st].quantile(.95)
            k=int(np.nansum(det)); n=len(c); lo,hi=wilson(k,n)
            if st=='GAP':
                # ARI de la partición ELEGIDA por la regla: k̂=1 -> partición única (ARI=0); k̂>=2 -> K-Means con ese k.
                # Fuente: det_eval (ARI_km_k{k}) o, para réplicas archivadas sin esa columna, el complemento verificado de s06b.
                kh=gap_khat(c,nul); av=[]
                for (ix,r),kk in zip(c.iterrows(),kh):
                    ctx=f'{e}/{ruido}/G{G}/δ{d}/réplica {int(r.replica)} (k̂={kk})'
                    col=f'ARI_km_k{kk}'
                    if kk==1: v,fu=0.0,'k1'
                    elif col in c.columns and not (isinstance(r.get(col),float) and np.isnan(r.get(col))):
                        # el archivo principal tiene un valor para ese k: debe ser válido (no se recurre al complemento)
                        fu='det_eval'; v=ari_valido(r.get(col),ctx+', fuente=det_eval')
                    else:
                        q=comp[(comp.escenario==e)&(comp.ruido==ruido)&(comp.G==G)&np.isclose(comp.delta.astype(float),d)&(comp.replica==r.replica)&(comp.k_gap==kk)]
                        if len(q)==0: sys.exit(f'ERROR: falta el ARI del gap para {ctx}; fuente=complemento; ejecutar s06b_completar_ari_gap.py')
                        if len(q)>1: sys.exit(f'ERROR: {len(q)} filas para {ctx} en el complemento (clave duplicada)')
                        fu='s06b_reconstruida'; v=ari_valido(q.ARI_gap.iloc[0],ctx+', fuente=gap_ari_complemento.csv')
                    av.append(v)
                    if nombre.startswith('G '): traza.append({'escenario':e,'ruido':ruido,'G':G,'delta':d,'replica':int(r.replica),'k_gap':int(kk),'ARI_gap':v,'fuente':fu})
                ari_m=float(np.mean(av)); ari_cov=1.0
            else: ari_m=c[arc].mean(); ari_cov=1.0
            filas.append({'escenario':e,'ruido':ruido,'procedimiento':nombre,'G':G,'delta':d,'n':n,'deteccion':k/n,'IC95_inf':lo,'IC95_sup':hi,
                          'ARI_medio':ari_m,'frac_ARI_disponible':ari_cov,'B_nulo':len(nul)})
T=pd.DataFrame(filas); TR=pd.DataFrame(traza)
# --- Validación de completitud ANTES de escribir cualquier salida definitiva ---
def falla(msg): sys.exit('ERROR de completitud: '+msg)
cond=TR.groupby(['escenario','ruido','G','delta'])
if cond.ngroups!=ev.groupby(['escenario','ruido','G','delta']).ngroups: falla(f'{cond.ngroups} condiciones del gap en la traza frente a {ev.groupby(["escenario","ruido","G","delta"]).ngroups} en det_eval')
if not (cond.size()==100).all(): falla('condiciones del gap sin exactamente 100 réplicas: '+str(cond.size()[cond.size()!=100].to_dict()))
if TR.duplicated(CLAVE).any(): falla('réplicas duplicadas en la traza del gap')
if not np.isfinite(TR.ARI_gap.astype(float)).all(): falla('ARI del gap no finito en la traza')
g=T[T.procedimiento.str.startswith('G ')]
if not (np.isfinite(g.ARI_medio).all() and (g.frac_ARI_disponible==1).all()): falla('media o cobertura del gap inválida')
def escribir(df,nombre):  # escritura atómica: nunca queda un archivo a medio escribir
    p=os.path.join(R_,nombre); df.to_csv(p+'.tmp',index=False); os.replace(p+'.tmp',p)
escribir(T,'tabla_potencia.csv'); escribir(TR,'gap_ari_por_replica.csv')
# --- Diagnóstico del porqué: tamaño mínimo de cluster en la solución de máximo silhouette ---
diag=[]
for e in ['E2','E3']:
    nul=pd.read_csv(os.path.join(R_,f'det_null_{e}_N0.csv')); c=ev[(ev.escenario==e)&(ev.ruido=='N0')]
    for lab,x in [('nulo',nul)]+[(f'G={G}, δ={d}',c[(c.G==G)&np.isclose(c.delta,d)]) for G in (2,3) for d in (0.3,0.5,0.8)]:
        diag.append({'escenario':e,'datos':lab,'frac_minsize<=2':(x.minsize_A_km<=2).mean(),'k_elegido_moda':int(x.kA_km.mode()[0]),
                     'frac_k=2':(x.kA_km==2).mean(),'sil_max_media':x.sil_max_km.mean(),'sil_k2_media':x.sil_km_k2.mean()})
D=pd.DataFrame(diag); D.to_csv(os.path.join(R_,'diagnostico_maxsil.csv'),index=False); print(D.round(3).to_string(index=False))
# --- Datos reales: todos los estadísticos, p frente a N0 (B=1000) y N2 (B=500) ---
reales=[]
for e in ['E2','E3']:
    s=escenario(e); o=stats(s[MESES].values.astype(float),rng=np.random.default_rng(7))
    for ruido in ['N0','N2','N2x2']:
        nul=pd.read_csv(os.path.join(R_,f'det_null_{e}_{ruido}.csv'))
        for nombre,(st,_) in PROC.items():
            if st in ('ORACULO',): continue
            if st=='GAP': val=int(gap_khat(pd.DataFrame([o]),nul)[0]); p=np.nan
            else: val=o[st]; p=(1+(nul[st]>=val).sum())/(1+len(nul))
            reales.append({'escenario':e,'ruido':ruido,'procedimiento':nombre,'valor':val,'p_empirico':p,'B':len(nul)})
RE=pd.DataFrame(reales); RE.to_csv(os.path.join(R_,'reales_todos_estadisticos.csv'),index=False)
print(RE[RE.ruido=='N0'].round(3).to_string(index=False))
# --- Resumen de potencia N0 ---
P0=T[T.ruido=='N0'].pivot_table(index=['procedimiento'],columns=['escenario','G','delta'],values='deteccion')
print(P0.round(2).to_string())
