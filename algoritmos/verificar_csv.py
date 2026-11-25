"""
verificar_csv.py
Control: compara los CSV de csv_originales/ (exportados de la base) con los
datos que usó la simulación (carpeta simulacion/datos). No modifica ningún archivo.

  1. Salida 13 de 2025 (cab. 2963)  frente a  datos/Resultados.csv
  2. Salida 16 de 2025 (cab. 2963)  frente a  los totales informados (4551 y 3941)
  3. Salida 13 de 2026 (cab. 4188)  frente a  datos/PTM_2025_2026_enero_mayo.csv (2026, ENE–MAY)

Uso (desde la carpeta algoritmos):  python verificar_csv.py
"""
from pathlib import Path
import pandas as pd

A_MANO = Path(__file__).resolve().parents[1]
CSV = A_MANO / 'csv_originales'
DATOS = A_MANO / 'simulacion' / 'datos'                 # datos de entrada de la simulación
MESES = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SET', 'OCT', 'NOV', 'DIC']


def leer(ruta):
    """Lee un CSV con «;», limpia los valores y separa código y nombre de la categoría."""
    tabla = pd.read_csv(ruta, sep=';', encoding='utf-8-sig', dtype=str)
    tabla = tabla.rename(columns={tabla.columns[1]: 'Clasificacion'})
    for col in MESES + ['TOTAL']:
        tabla[col] = tabla[col].str.strip().replace({'-': '0', '': '0'}).astype(int)
    if tabla['Clasificacion'].str.contains(r'\*').all():
        tabla['codigo'] = tabla['Clasificacion'].str.split('*', n=1).str[0].astype(int)
        tabla = tabla.set_index('codigo').sort_index()
    return tabla


# 1) Salida 13 de 2025
nueva = leer(CSV / 'salida13_organo_tejido_2025_cab2963.csv')
analisis = leer(DATOS / 'Resultados.csv')
iguales = nueva.index.equals(analisis.index)
distintas = int((nueva[MESES + ['TOTAL']] != analisis[MESES + ['TOTAL']]).sum().sum()) if iguales else 'códigos distintos'
print('1) Salida 13, 2025:', len(nueva), 'categorías,', nueva['TOTAL'].sum(), 'conteos;',
      'celdas distintas de Resultados.csv:', distintas)

# 2) Salida 16 de 2025 (sexo)
sexo = leer(CSV / 'salida16_sexo_2025_cab2963.csv').set_index('Clasificacion')
print('2) Salida 16, 2025:', {k: int(v) for k, v in sexo['TOTAL'].items()},
      '| informado: MASCULINO 4551, FEMENINO 3941')
print('   Totales mensuales por sexo = totales mensuales de la salida 13:',
      bool((sexo[MESES].sum() == nueva[MESES].sum()).all()))

# 3) Salida 13 de 2026, enero a mayo
n26 = leer(CSV / 'salida13_organo_tejido_2026_cab4188.csv')
t = pd.read_csv(DATOS / 'PTM_2025_2026_enero_mayo.csv', encoding='utf-8-sig')
t26 = t[t['anio'] == 2026].set_index('codigo').sort_index()
ene_may = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY']
comunes = n26.index.intersection(t26.index)
dif = int((n26.loc[comunes, ene_may].values != t26.loc[comunes, [m.lower() for m in ene_may]].values).sum())
print('3) Salida 13, 2026:', len(n26), 'categorías; en el CSV temporal:', len(t26),
      '| comunes:', len(comunes), '| celdas ENE–MAY distintas:', dif)
