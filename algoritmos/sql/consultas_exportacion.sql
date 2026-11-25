/* =============================================================================
   consultas_exportacion.sql
   Consultas ejecutadas en SQL Server Management Studio el 09/11/2026 para
   exportar los CSV de a mano/csv_originales/.

   Servidor: .\SQLEXPRESS   Base: Incucai   Autenticación: Windows
   Procedimiento almacenado del equipo: sp_get_table_PacienteTransplantadoMensual
     @IdCabecera = descarga del reporte CRESI guardada en la base
     @IdTabla    = número de salida del reporte (13 = órgano o tejido, 16 = sexo)

   Exportación: grilla de resultados → seleccionar todo → «Copy with Headers»
   → pegar en Excel → guardar como CSV (separador «;»).
   Las consultas solo leen la base.
   ============================================================================= */

USE Incucai;

-- 0) Qué son las descargas usadas: año, tipo de donante (0 = FALLECIDOS),
--    período (2 = fecha de desclampeo del receptor) y fecha de descarga.
SELECT * FROM PacienteTransplantadoMensual WHERE IdCabecera IN (2963, 4188);

-- 1) Salida 13 de 2025, descarga del 31/12/2025 18:01
--    → csv_originales/salida13_organo_tejido_2025_cab2963.csv  (78 categorías, total 8.492)
EXEC sp_get_table_PacienteTransplantadoMensual @IdCabecera = 2963, @IdTabla = 13;

-- 2) Salida 16 de 2025 (por sexo), misma descarga
--    → csv_originales/salida16_sexo_2025_cab2963.csv  (MASCULINO 4.551, FEMENINO 3.941)
EXEC sp_get_table_PacienteTransplantadoMensual @IdCabecera = 2963, @IdTabla = 16;

-- 3) Salida 13 de 2026, descarga del 13/06/2026 18:01 (junio incompleto)
--    → csv_originales/salida13_organo_tejido_2026_cab4188.csv  (69 categorías)
EXEC sp_get_table_PacienteTransplantadoMensual @IdCabecera = 4188, @IdTabla = 13;
