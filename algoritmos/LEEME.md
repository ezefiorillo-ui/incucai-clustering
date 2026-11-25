# Algoritmos del paper

Cada figura se sigue hacia atrás así: **origen → CSV → script → figura**.

```text
(raíz del repositorio)
├── csv_originales/         DATOS: solo los CSV exportados de la base con SQL
├── resultados_simulacion/  tablas de la simulación (no vienen de SQL)
├── resultados_nuevos/      salidas de analisis_E5_CLR_DBSCAN.py (Tabla 5)
├── algoritmos/             SCRIPTS (esta carpeta)
│   └── sql/                consultas que generaron los CSV
├── figuras/                lo que generan los scripts y se pega en el Word
└── simulacion/             código y resultados de la simulación
```

## Cadena de cada figura y tabla

Los nombres de los archivos no coinciden con la numeración del artículo: la Fig. 1 del artículo es `fig2_mapa_calor_E2.png` y la Fig. 2 es `fig1_diseno.png`.

| Resultado en el artículo | Origen | CSV de entrada | Script | Salida |
|---|---|---|---|---|
| Fig. 1, perfiles mensuales de E2 (mapa de calor) | Base Incucai, cabecera 2963 | `salida13_organo_tejido_2025_cab2963.csv` | `fig2_mapa_calor_E2.py` | `figuras/fig2_mapa_calor_E2.png` y `fig2_valores.csv` |
| Fig. 2, esquema del diseño | No usa datos | — | — (dibujo propio) | `figuras/fig1_diseno.png` |
| Fig. 3, ARI por algoritmo (control positivo) | Simulación `s03_control_positivo.py` → `s04_consolidar.py` (carpeta `simulacion/scripts/`) | `resultados_simulacion/tabla_control_positivo.csv` | `fig3_ARI_algoritmos.py` | `figuras/fig3_ARI_algoritmos.png` |
| Fig. 4, criterios de detección | Simulación `s06_deteccion.py` → `s07_consolidar_deteccion.py` | `resultados_simulacion/tabla_potencia.csv` | `fig4_deteccion.py` | `figuras/fig4_deteccion.png` |
| Tabla 4, datos reales frente a N0 | Simulación `s06_deteccion.py` → `s07_consolidar_deteccion.py` (K-Means, Ward, R2 y Z) y `s02_nulos.py` (GMM) | `simulacion/resultados/reales_todos_estadisticos.csv` y `pvalores_nulos.csv` | — (valores copiados de los CSV) | Tabla 4 del artículo |
| Sección 5.4, corrección de Holm y magnitud de Z | Simulación | `reales_todos_estadisticos.csv` y `det_null_E2_N0.csv` | `holm_y_magnitud_Z.py` | Informe en pantalla (0,022 y 0,084; 0,273, 0,171 y 0,205) |
| Sección 4.3, mínimo de 3 categorías de DBSCAN | Base Incucai, cabecera 2963 | `simulacion/datos/categorias_78_clasificadas.csv` | `dbscan_min_samples.py` | Informe en pantalla (con un mínimo de 13, todo ruido en los radios chicos) |
| Tabla 5, E5, CLR y ruido de DBSCAN | Base Incucai, cabecera 2963 | `salida13_organo_tejido_2025_cab2963.csv` | `analisis_E5_CLR_DBSCAN.py` | `resultados_nuevos/` |
| Sección 5.4, Z sin SISTEMA OSTEOARTICULAR en E2 | Base Incucai, cabecera 2963 | `salida13_organo_tejido_2025_cab2963.csv` | `z_E2_sin_osteoarticular.py` | Informe en pantalla (p = 0,326) |
| Secciones 5.2 y 5.3, intervalos de confianza | Simulación | `simulacion/resultados/control_positivo.csv` y `tabla_potencia.csv` | `ic_controles_positivos.py` | Informe en pantalla |
| Control de datos | Base Incucai y datos de la simulación | Los tres CSV de la base y `simulacion/datos/` | `verificar_csv.py` | Informe en pantalla |

## Cómo se obtuvieron los CSV

- **Desde la base** (`salida13_…_cab2963`, `salida16_…_cab2963`, `salida13_…_cab4188`): con las consultas de `sql/consultas_exportacion.sql`, ejecutadas en Management Studio y guardadas como CSV.
- **Desde la simulación** (`resultados_simulacion/`): copias de los resultados archivados en `simulacion/resultados/`.
  - `tabla_control_positivo.csv`: comparación de K-Means, Ward, GMM y DBSCAN, generada por `simulacion/scripts/s03_control_positivo.py` y `s04_consolidar.py`.
  - `tabla_potencia.csv`: tasas de detección, generada por `simulacion/scripts/s06_deteccion.py` y `s07_consolidar_deteccion.py`. Detección = el estadístico supera el percentil 95 de su distribución bajo el nulo; para el gap, elegir k ≥ 2.

## Cómo ejecutar

Abrir una terminal en esta carpeta y ejecutar:

```
python verificar_csv.py
python fig2_mapa_calor_E2.py
python fig3_ARI_algoritmos.py
python fig4_deteccion.py
python analisis_E5_CLR_DBSCAN.py
```

**Qué tiene que salir:**
- `verificar_csv.py`: 0 celdas distintas, MASCULINO 4551 y FEMENINO 3941, True, y 0 celdas distintas en 2026.
- `fig2_mapa_calor_E2.py`: «Categorías en el archivo: 78 | conteos: 8492», luego «Categorías en E2: 22 | conteos: 7830» y la figura guardada en `figuras/`.
- `fig3_ARI_algoritmos.py`: para E2, G = 2, δ = 0,3, muestra KMeans 0.775, Ward 0.731, GMM 0.617 y DBSCAN 0.247 (en el paper: entre 0,62 y 0,78 para los tres primeros).
- `fig4_deteccion.py`: para E2, G = 2, δ = 0,3, muestra «B silhouette k verdadero» 0.00 y «R reducción de dispersión K-Means k=2» 0.96 (en el paper: 0/100 y 96/100).
- `analisis_E5_CLR_DBSCAN.py`: dos tablas, una de p empíricos (Tabla 5) y otra del ruido de DBSCAN. Tarda unos 25 minutos; para una prueba rápida, `python analisis_E5_CLR_DBSCAN.py 20`.

## Funciones usadas (para la sección de reproducibilidad)

- `pandas.read_csv`: lee los CSV.
- `DataFrame.div`: divide cada fila por su total anual, p(i, m) = x(i, m) / T(i).
- `matplotlib.pyplot.imshow`: dibuja el mapa de calor.
- `Figure.colorbar`: agrega la barra de color.
- `Figure.savefig`: guarda la figura a 300 dpi.
