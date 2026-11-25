# Evaluación comparativa de algoritmos de clustering y detección de estructura en perfiles mensuales de órganos y tejidos (INCUCAI/CRESI)

Código, datos y resultados del trabajo de Roxana Martínez y Ezequiel Fiorillo (CAETI, Universidad Abierta Interamericana).

El trabajo evalúa si los perfiles mensuales de órganos y tejidos implantados en 2025, según CRESI/INCUCAI, tienen grupos naturales. Para eso compara cuatro algoritmos (K-Means, Ward, GMM y DBSCAN) contra modelos nulos, es decir, datos simulados sin grupos, y contra controles positivos, es decir, datos con grupos conocidos.

## Datos

- **Fuente:** CRESI, «Registro nacional de procuración y trasplantes» → «Reporte de pacientes trasplantados comparativo mensual».
- **Filtros:** período del 01/01/2025 al 31/12/2025, por fecha de desclampeo del receptor; donante fallecido; salida 13 (por órgano o tejido) y salida 16 (por sexo).
- **Versión:** descarga del 31/12/2025 a las 18:01, con 78 categorías y 8492 registros. CRESI actualiza registros de fechas anteriores, así que una consulta posterior puede dar otros valores.
- Los datos son públicos y agregados: no contienen información personal.

## Estructura

```text
csv_originales/         CSV exportados de la base (salidas 13 y 16 de 2025, y salida 13 de 2026)
algoritmos/             scripts de las figuras, del análisis E5/CLR/DBSCAN y de verificación de los datos
  sql/                  consultas con las que se exportaron los CSV
figuras/                figuras del artículo
resultados_simulacion/  tablas de la simulación que usan las figuras 3 y 4
resultados_nuevos/      resultados de E5, CLR y ruido de DBSCAN (Tabla 5)
simulacion/             modelos nulos (N0, N1, N2), controles positivos y detección
  scripts/              código de la simulación
  datos/                tabla de entrada de la simulación
  resultados/           resultados archivados (p empíricos, tasas de detección, ARI)
```

## Entorno

Python 3.12, numpy, pandas, scipy, matplotlib y scikit-learn 1.8.0. Para instalar las dependencias:

```
pip install -r requirements.txt
```

En Windows, antes de ejecutar los scripts de `simulacion/`, activar el modo UTF-8 para que la consola pueda mostrar letras griegas como δ y ρ: `$env:PYTHONUTF8=1` en PowerShell, o `set PYTHONUTF8=1` en cmd.

## Figuras del artículo

Los nombres de los archivos no siguen la numeración del artículo:

| Figura del artículo | Archivo |
|---|---|
| Fig. 1, perfiles mensuales de las 22 categorías de E2 | `figuras/fig2_mapa_calor_E2.png` |
| Fig. 2, diseño experimental | `figuras/fig1_diseno.png` |
| Fig. 3, ARI medio de cada algoritmo | `figuras/fig3_ARI_algoritmos.png` |
| Fig. 4, detección de grupos de cada criterio | `figuras/fig4_deteccion.png` |

## Cómo reproducir

**Figuras y verificación de los datos** (desde `algoritmos/`):

```
python verificar_csv.py
python fig2_mapa_calor_E2.py
python fig3_ARI_algoritmos.py
python fig4_deteccion.py
```

El detalle de qué tiene que mostrar cada script está en `algoritmos/LEEME.md`.

**E5, CLR y ruido de DBSCAN (Tabla 5)** (desde `algoritmos/`):

```
python analisis_E5_CLR_DBSCAN.py
```

Usa 1000 simulaciones por escenario y tarda unos 25 minutos. Para una prueba rápida: `python analisis_E5_CLR_DBSCAN.py 20`. Este script también recalcula E2 y E3 con proporciones, pero con otras simulaciones, así que sus p difieren levemente de los de la Tabla 4 (por ejemplo, Ward en E3: 0,087 en lugar de 0,083).

**De dónde sale la Tabla 4:** las filas de K-Means, Ward, R2 y Z salen de `simulacion/resultados/reales_todos_estadisticos.csv` (filas `N0`), y la fila de GMM, de `simulacion/resultados/pvalores_nulos.csv`. Los p de N1 y N2 que se citan en la sección 5.4 (Ward en E3: 0,057 y 0,074) también salen de `pvalores_nulos.csv`.

**Mínimo de categorías de DBSCAN (sección 4.3)** (desde `algoritmos/`): `python dbscan_min_samples.py`. Con un mínimo de 13, en los dos radios más chicos de E2 todas las categorías quedan como ruido; con 3 o 5, forma un solo grupo.

**Corrección de Holm y magnitud de Z (sección 5.4)** (desde `algoritmos/`):

```
python holm_y_magnitud_Z.py
```

Muestra las 22 pruebas de la familia con su p ajustado (Z en E2: 0,022; Z en E3: 0,084) y la magnitud de Z en E2 (real 0,273; mediana de N0 0,171; percentil 95 0,205).

**Z en E2 sin SISTEMA OSTEOARTICULAR (sección 5.4)** (desde `algoritmos/`):

```
python z_E2_sin_osteoarticular.py
```

Resultado esperado: Z = 0,1835 y p = 0,326.

**Intervalos de confianza de las secciones 5.2 y 5.3** (desde `algoritmos/`):

```
python ic_controles_positivos.py
```

Muestra el intervalo del 95 % del ARI medio y de la tasa de detección (intervalo de Wilson) para dos grupos con δ = 0,3.

**Versiones:** la simulación se ejecutó con Python 3.12 y scikit-learn 1.8.0. `analisis_E5_CLR_DBSCAN.py` y `z_E2_sin_osteoarticular.py` se ejecutaron con Python 3.14 y scikit-learn 1.9.1.

**Simulación completa** (desde `simulacion/scripts/`):

| Paso | Comando | Qué produce |
|---|---|---|
| 1 | `python s01_sobredispersion.py` | estimación de ρ para N1 y N2 |
| 2 | `python s02_nulos.py E2 N0 1000` (también E3, N1 y N2) | modelos nulos y p empíricos (Tabla 4 y N1/N2) |
| 3 | `python s03_control_positivo.py E2 2 100` (también E2 3, E3 2 y E3 3) | controles positivos |
| 4 | `python s04_consolidar.py` | tabla de controles positivos (Fig. 3) |
| 5 | `python s06_deteccion.py null E2 N0 1000` y `python s06_deteccion.py eval E2 N0 100 2,3 0,0.2,0.3,0.5,0.8` (también E3) | distribuciones nulas y detección |
| 6 | `python s06b_completar_ari_gap.py` | completa el ARI del gap |
| 7 | `python s08_estabilidad.py` | estabilidad de Ward |
| 8 | `python s07_consolidar_deteccion.py` y `python s09_figuras_deteccion.py` | tasas de detección (Fig. 4) y estadísticos de los datos reales |

Los archivos con `N1x2` y `N2x2` son análisis de sensibilidad extra, con el doble de variación que N1 y N2. No se informan en el artículo.

## Parámetros

- **K-Means:** k de 2 a 6, 20 inicializaciones, `random_state=42`.
- **Ward:** k de 2 a 6, enlace de Ward con distancia euclídea.
- **GMM:** k de 2 a 6, covarianza esférica, 3 inicializaciones, tolerancia 0,001, máximo 100 iteraciones, `random_state=42`.
- **DBSCAN:** `min_samples=3`, cinco radios iguales a los percentiles 10, 20, 30, 40 y 50 de las distancias entre categorías.
- **Selección:** en cada algoritmo, la partición de mayor silhouette, con el mismo procedimiento en los datos reales y en los simulados.
- **p empírico:** (1 + E) / (B + 1), donde E es la cantidad de simulaciones con un valor igual o mayor al observado.
- **Semilla global de la simulación:** 20261017.
