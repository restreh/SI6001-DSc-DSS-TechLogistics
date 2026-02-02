"""
Módulo de Limpieza y Curación de Datos
TechLogistics S.A. - Sistema de Soporte a la Decisión (DSS)

Este módulo contiene todas las funciones necesarias para:
- Auditoría de calidad de datos
- Limpieza y normalización
- Detección y tratamiento de outliers
- Generación de reportes de salud de datos
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re


# =============================================================================
# DICCIONARIOS DE NORMALIZACIÓN
# =============================================================================

CIUDAD_MAPPING = {
    'med': 'Medellín',
    'medellin': 'Medellín',
    'medellín': 'Medellín',
    'bog': 'Bogotá',
    'bogota': 'Bogotá',
    'bogotá': 'Bogotá',
    'cali': 'Cali',
    'barranquilla': 'Barranquilla',
    'bucaramanga': 'Bucaramanga',
    'ventas_web': 'Canal Digital',
    'cartagena': 'Cartagena',
    'pereira': 'Pereira',
    'manizales': 'Manizales',
    'santa marta': 'Santa Marta',
    'ibague': 'Ibagué',
    'ibagué': 'Ibagué',
    'cucuta': 'Cúcuta',
    'cúcuta': 'Cúcuta'
}

BODEGA_MAPPING = {
    'norte': 'Norte',
    'sur': 'Sur',
    'occidente': 'Occidente',
    'oriente': 'Oriente',
    'zona_franca': 'Zona Franca',
    'bod-ext-99': 'Bodega Externa',
    'centro': 'Centro'
}

CATEGORIA_MAPPING = {
    'smart-phone': 'Smartphones',
    'smartphones': 'Smartphones',
    'smartphone': 'Smartphones',
    'accesorios': 'Accesorios',
    'monitores': 'Monitores',
    'tablets': 'Tablets',
    'laptops': 'Laptops',
    'audio': 'Audio',
    'gaming': 'Gaming',
    'wearables': 'Wearables',
    '???': 'Sin Categoría',
    'nan': 'Sin Categoría',
    '': 'Sin Categoría'
}

TICKET_MAPPING = {
    'sí': True, 'si': True, '1': True, 1: True, 'yes': True, True: True,
    'no': False, '0': False, 0: False, 'false': False, False: False,
    'n/a': None, 'nan': None, '': None
}

RECOMIENDA_MAPPING = {
    'sí': 'Sí', 'si': 'Sí', 'yes': 'Sí',
    'no': 'No', 'false': 'No',
    'maybe': 'Tal vez', 'quizas': 'Tal vez',
    'n/a': 'Sin respuesta', 'nan': 'Sin respuesta', '': 'Sin respuesta'
}


# =============================================================================
# FUNCIONES DE MÉTRICAS DE CALIDAD
# =============================================================================

def calculate_health_score(df, dataset_name="Dataset"):
    """
    Calcula un Health Score para el dataset basado en múltiples métricas.
    
    Métricas consideradas:
    - Completitud: % de valores no nulos
    - Unicidad: % de registros no duplicados
    - Validez: % de valores dentro de rangos esperados
    
    Returns:
        dict: Diccionario con todas las métricas de salud
    """
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    
    # Completitud (peso 40%)
    completitud = ((total_cells - null_cells) / total_cells) * 100
    
    # Unicidad (peso 30%)
    unicidad = ((df.shape[0] - duplicates) / df.shape[0]) * 100
    
    # Validez base (peso 30%) - se ajusta según el dataset
    validez = 100.0  # Se ajustará según validaciones específicas
    
    # Health Score ponderado
    health_score = (completitud * 0.4) + (unicidad * 0.3) + (validez * 0.3)
    
    # Detalle de nulidad por columna
    null_by_column = df.isnull().sum()
    null_pct_by_column = (null_by_column / len(df) * 100).round(2)
    
    return {
        'dataset_name': dataset_name,
        'total_registros': df.shape[0],
        'total_columnas': df.shape[1],
        'total_celdas': total_cells,
        'celdas_nulas': int(null_cells),
        'registros_duplicados': int(duplicates),
        'completitud_pct': round(completitud, 2),
        'unicidad_pct': round(unicidad, 2),
        'validez_pct': round(validez, 2),
        'health_score': round(health_score, 2),
        'nulidad_por_columna': null_pct_by_column.to_dict(),
        'columnas_con_nulos': null_by_column[null_by_column > 0].to_dict()
    }


def detect_outliers_iqr(series, multiplier=1.5):
    """
    Detecta outliers usando el método del rango intercuartílico (IQR).
    
    Args:
        series: Serie de pandas con valores numéricos
        multiplier: Multiplicador para el IQR (default 1.5)
    
    Returns:
        tuple: (máscara booleana de outliers, límite inferior, límite superior)
    """
    if series.dtype not in ['int64', 'float64']:
        return pd.Series([False] * len(series)), None, None
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    
    outliers_mask = (series < lower_bound) | (series > upper_bound)
    
    return outliers_mask, lower_bound, upper_bound


def generate_outlier_report(df, numeric_columns):
    """
    Genera un reporte detallado de outliers para columnas numéricas.
    """
    report = {}
    for col in numeric_columns:
        if col in df.columns:
            mask, lower, upper = detect_outliers_iqr(df[col].dropna())
            n_outliers = mask.sum()
            if n_outliers > 0:
                report[col] = {
                    'cantidad_outliers': int(n_outliers),
                    'porcentaje': round(n_outliers / len(df) * 100, 2),
                    'limite_inferior': round(lower, 2) if lower else None,
                    'limite_superior': round(upper, 2) if upper else None,
                    'min_valor': round(df[col].min(), 2),
                    'max_valor': round(df[col].max(), 2)
                }
    return report


# =============================================================================
# FUNCIONES DE LIMPIEZA POR DATASET
# =============================================================================

def clean_inventario(df):
    """
    Limpia y normaliza el dataset de inventario.
    
    Tratamientos:
    - Normalización de categorías
    - Normalización de bodegas
    - Conversión de Lead_Time_Dias a numérico
    - Tratamiento de Stock negativo
    - Detección de costos anómalos
    """
    df_clean = df.copy()
    cleaning_log = {
        'registros_originales': len(df),
        'acciones': [],
        'outliers_detectados': {},
        'imputaciones': {},
        'outliers_dataframes': {}  # Guardar DataFrames originales de outliers
    }
    
    # 1. Eliminar duplicados exactos
    duplicados_antes = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    if duplicados_antes > 0:
        cleaning_log['acciones'].append(f"Eliminados {duplicados_antes} registros duplicados exactos")
    
    # 2. Normalizar Categoria
    df_clean['Categoria_Original'] = df_clean['Categoria']
    df_clean['Categoria'] = df_clean['Categoria'].apply(
        lambda x: CATEGORIA_MAPPING.get(str(x).lower().strip(), 'Sin Categoría') 
        if pd.notna(x) else 'Sin Categoría'
    )
    cleaning_log['acciones'].append("Normalización de categorías completada")
    
    # 3. Normalizar Bodega_Origen
    df_clean['Bodega_Original'] = df_clean['Bodega_Origen']
    df_clean['Bodega_Origen'] = df_clean['Bodega_Origen'].apply(
        lambda x: BODEGA_MAPPING.get(str(x).lower().strip(), str(x).title()) 
        if pd.notna(x) else 'Desconocida'
    )
    cleaning_log['acciones'].append("Normalización de bodegas completada")
    
    # 4. Convertir Lead_Time_Dias a numérico
    # Guardar valor original antes de convertir
    df_clean['Lead_Time_Original'] = df.loc[df_clean.index, 'Lead_Time_Dias'] if 'Lead_Time_Dias' in df.columns else None
    
    def parse_lead_time(val):
        if pd.isna(val) or str(val).lower() in ['nan', 'none', '']:
            return np.nan
        val_str = str(val).lower().strip()
        if val_str == 'inmediato':
            return 1
        if '-' in val_str:
            # Formato "25-30 días" -> tomar promedio
            nums = re.findall(r'\d+', val_str)
            if len(nums) >= 2:
                return (int(nums[0]) + int(nums[1])) / 2
        try:
            return float(val_str)
        except:
            return np.nan
    
    df_clean['Lead_Time_Dias'] = df_clean['Lead_Time_Original'].apply(parse_lead_time)
    
    # Imputar Lead_Time nulos con la mediana por categoría
    lead_time_mediana = df_clean.groupby('Categoria')['Lead_Time_Dias'].transform('median')
    nulos_lead_time = df_clean['Lead_Time_Dias'].isna().sum()
    df_clean['Lead_Time_Dias'] = df_clean['Lead_Time_Dias'].fillna(lead_time_mediana)
    df_clean['Lead_Time_Dias'] = df_clean['Lead_Time_Dias'].fillna(df_clean['Lead_Time_Dias'].median())
    cleaning_log['imputaciones']['Lead_Time_Dias'] = {
        'metodo': 'Mediana por categoría',
        'justificacion': 'La mediana es robusta a outliers y el agrupamiento por categoría refleja patrones de negocio',
        'valores_imputados': int(nulos_lead_time)
    }
    
    # 5. Tratar Stock negativo - GUARDAR ORIGINALES ANTES DE CORREGIR
    df_clean['Stock_Original'] = df_clean['Stock_Actual'].copy()
    stocks_negativos_mask = df_clean['Stock_Actual'] < 0
    df_clean['Stock_Negativo_Flag'] = stocks_negativos_mask
    
    # Guardar DataFrame original de stocks negativos ANTES de corregir
    if stocks_negativos_mask.sum() > 0:
        cleaning_log['outliers_dataframes']['stock_negativo'] = df_clean[stocks_negativos_mask][
            ['SKU_ID', 'Categoria', 'Stock_Original', 'Bodega_Origen', 'Costo_Unitario_USD']
        ].copy()
    
    df_clean.loc[df_clean['Stock_Actual'] < 0, 'Stock_Actual'] = 0
    cleaning_log['acciones'].append(f"Corregidos {stocks_negativos_mask.sum()} registros con stock negativo (establecidos a 0)")
    
    # Imputar Stock nulos con 0 (asumiendo quiebre de stock)
    nulos_stock = df_clean['Stock_Actual'].isna().sum()
    df_clean['Stock_Actual'] = df_clean['Stock_Actual'].fillna(0)
    cleaning_log['imputaciones']['Stock_Actual'] = {
        'metodo': 'Valor cero',
        'justificacion': 'Stock nulo se interpreta como quiebre de stock, consistente con la lógica de inventario',
        'valores_imputados': int(nulos_stock)
    }
    
    # 6. Detectar costos anómalos (outliers) - GUARDAR ORIGINALES
    mask_outliers, lower, upper = detect_outliers_iqr(df_clean['Costo_Unitario_USD'], multiplier=3)
    df_clean['Costo_Outlier_Flag'] = mask_outliers
    
    # Guardar DataFrame original de outliers de costo
    if mask_outliers.sum() > 0:
        cleaning_log['outliers_dataframes']['costo_outliers'] = df_clean[mask_outliers][
            ['SKU_ID', 'Categoria', 'Costo_Unitario_USD', 'Stock_Actual', 'Bodega_Origen']
        ].copy()
    
    cleaning_log['outliers_detectados']['Costo_Unitario_USD'] = {
        'cantidad': int(mask_outliers.sum()),
        'limite_inferior': round(lower, 2) if lower else None,
        'limite_superior': round(upper, 2) if upper else None
    }
    
    # 7. Convertir Ultima_Revision a datetime
    df_clean['Ultima_Revision'] = pd.to_datetime(df_clean['Ultima_Revision'], errors='coerce')
    
    # Detectar fechas futuras
    fecha_actual = datetime.now()
    fechas_futuras = (df_clean['Ultima_Revision'] > fecha_actual).sum()
    if fechas_futuras > 0:
        cleaning_log['acciones'].append(f"Detectadas {fechas_futuras} fechas de revisión futuras")
    
    # Calcular días desde última revisión
    df_clean['Dias_Sin_Revision'] = (fecha_actual - df_clean['Ultima_Revision']).dt.days
    df_clean.loc[df_clean['Dias_Sin_Revision'] < 0, 'Dias_Sin_Revision'] = 0
    
    cleaning_log['registros_finales'] = len(df_clean)
    
    return df_clean, cleaning_log


def clean_transacciones(df):
    """
    Limpia y normaliza el dataset de transacciones.
    
    Tratamientos:
    - Normalización de ciudades
    - Conversión de fechas
    - Tratamiento de cantidades negativas
    - Detección de tiempos de entrega anómalos
    """
    df_clean = df.copy()
    cleaning_log = {
        'registros_originales': len(df),
        'acciones': [],
        'outliers_detectados': {},
        'imputaciones': {},
        'outliers_dataframes': {}  # Guardar DataFrames originales de outliers
    }
    
    # 1. Eliminar duplicados exactos
    duplicados_antes = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    if duplicados_antes > 0:
        cleaning_log['acciones'].append(f"Eliminados {duplicados_antes} registros duplicados exactos")
    
    # 2. Normalizar Ciudad_Destino
    df_clean['Ciudad_Original'] = df_clean['Ciudad_Destino']
    df_clean['Ciudad_Destino'] = df_clean['Ciudad_Destino'].apply(
        lambda x: CIUDAD_MAPPING.get(str(x).lower().strip(), str(x).title()) 
        if pd.notna(x) else 'Desconocida'
    )
    cleaning_log['acciones'].append("Normalización de ciudades completada")
    
    # 3. Convertir Fecha_Venta a datetime
    # Guardar fecha original como string
    df_clean['Fecha_Venta_Original'] = df.loc[df_clean.index, 'Fecha_Venta'].copy() if 'Fecha_Venta' in df.columns else None
    df_clean['Fecha_Venta'] = pd.to_datetime(df_clean['Fecha_Venta_Original'], format='%d/%m/%Y', errors='coerce')
    
    # Validar fechas (no futuras más allá de hoy)
    fecha_actual = datetime.now()
    fechas_futuras_mask = df_clean['Fecha_Venta'] > fecha_actual
    df_clean['Fecha_Futura_Flag'] = fechas_futuras_mask
    
    # Guardar DataFrame original de fechas futuras
    if fechas_futuras_mask.sum() > 0:
        cleaning_log['outliers_dataframes']['fechas_futuras'] = df_clean[fechas_futuras_mask][
            ['Transaccion_ID', 'SKU_ID', 'Fecha_Venta', 'Fecha_Venta_Original', 'Cantidad_Vendida', 'Precio_Venta_Final', 'Canal_Venta']
        ].copy()
    
    cleaning_log['acciones'].append(f"Detectadas {fechas_futuras_mask.sum()} transacciones con fecha futura")
    
    # 4. Tratar cantidades negativas - GUARDAR ORIGINALES ANTES DE CORREGIR
    df_clean['Cantidad_Original'] = df_clean['Cantidad_Vendida'].copy()
    cantidades_negativas_mask = df_clean['Cantidad_Vendida'] < 0
    df_clean['Cantidad_Negativa_Flag'] = cantidades_negativas_mask
    
    # Guardar DataFrame original de cantidades negativas ANTES de corregir
    if cantidades_negativas_mask.sum() > 0:
        cleaning_log['outliers_dataframes']['cantidades_negativas'] = df_clean[cantidades_negativas_mask][
            ['Transaccion_ID', 'SKU_ID', 'Cantidad_Original', 'Precio_Venta_Final', 'Canal_Venta', 'Fecha_Venta']
        ].copy()
    
    df_clean.loc[df_clean['Cantidad_Vendida'] < 0, 'Cantidad_Vendida'] = \
        df_clean.loc[df_clean['Cantidad_Vendida'] < 0, 'Cantidad_Vendida'].abs()
    cleaning_log['acciones'].append(f"Corregidas {cantidades_negativas_mask.sum()} cantidades negativas (convertidas a valor absoluto)")
    
    # 5. Detectar y tratar tiempos de entrega anómalos - GUARDAR ORIGINALES
    df_clean['Tiempo_Entrega_Original'] = df_clean['Tiempo_Entrega_Real'].copy()
    mask_tiempo, lower_t, upper_t = detect_outliers_iqr(df_clean['Tiempo_Entrega_Real'], multiplier=3)
    df_clean['Tiempo_Entrega_Outlier_Flag'] = mask_tiempo
    
    # Guardar DataFrame original de outliers de tiempo ANTES de corregir
    if mask_tiempo.sum() > 0:
        cleaning_log['outliers_dataframes']['tiempo_entrega_outliers'] = df_clean[mask_tiempo][
            ['Transaccion_ID', 'SKU_ID', 'Tiempo_Entrega_Original', 'Ciudad_Destino', 'Estado_Envio', 'Canal_Venta']
        ].copy()
    
    # Capear tiempos extremos a un máximo razonable (90 días)
    tiempos_extremos = (df_clean['Tiempo_Entrega_Real'] > 90).sum()
    df_clean.loc[df_clean['Tiempo_Entrega_Real'] > 90, 'Tiempo_Entrega_Real'] = 90
    
    cleaning_log['outliers_detectados']['Tiempo_Entrega_Real'] = {
        'cantidad': int(mask_tiempo.sum()),
        'limite_superior_iqr': round(upper_t, 2) if upper_t else None,
        'valores_capeados_90dias': int(tiempos_extremos)
    }
    
    # 6. Imputar Costo_Envio nulos con la mediana por ciudad
    costo_envio_mediana = df_clean.groupby('Ciudad_Destino')['Costo_Envio'].transform('median')
    nulos_costo = df_clean['Costo_Envio'].isna().sum()
    df_clean['Costo_Envio'] = df_clean['Costo_Envio'].fillna(costo_envio_mediana)
    df_clean['Costo_Envio'] = df_clean['Costo_Envio'].fillna(df_clean['Costo_Envio'].median())
    cleaning_log['imputaciones']['Costo_Envio'] = {
        'metodo': 'Mediana por ciudad destino',
        'justificacion': 'Los costos de envío varían por zona geográfica, la mediana por ciudad captura esta variabilidad',
        'valores_imputados': int(nulos_costo)
    }
    
    # 7. Normalizar Estado_Envio
    df_clean['Estado_Envio'] = df_clean['Estado_Envio'].fillna('Sin Información')
    df_clean['Estado_Envio'] = df_clean['Estado_Envio'].str.strip().str.title()
    
    # Extraer componentes temporales
    df_clean['Año_Venta'] = df_clean['Fecha_Venta'].dt.year
    df_clean['Mes_Venta'] = df_clean['Fecha_Venta'].dt.month
    df_clean['Dia_Semana'] = df_clean['Fecha_Venta'].dt.dayofweek
    
    cleaning_log['registros_finales'] = len(df_clean)
    
    return df_clean, cleaning_log


def clean_feedback(df):
    """
    Limpia y normaliza el dataset de feedback de clientes.
    
    Tratamientos:
    - Eliminación de duplicados
    - Normalización de respuestas categóricas
    - Tratamiento de edades imposibles
    - Normalización de NPS
    - Detección de ratings anómalos
    """
    df_clean = df.copy()
    cleaning_log = {
        'registros_originales': len(df),
        'acciones': [],
        'outliers_detectados': {},
        'imputaciones': {},
        'outliers_dataframes': {}  # Guardar DataFrames originales de outliers
    }
    
    # 1. Eliminar duplicados exactos
    duplicados_antes = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    
    # También eliminar duplicados por Feedback_ID (clave primaria)
    duplicados_id = df_clean.duplicated(subset=['Feedback_ID']).sum()
    df_clean = df_clean.drop_duplicates(subset=['Feedback_ID'], keep='first')
    
    total_duplicados = duplicados_antes + duplicados_id
    if total_duplicados > 0:
        cleaning_log['acciones'].append(f"Eliminados {total_duplicados} registros duplicados")
    
    # 2. Normalizar Ticket_Soporte_Abierto
    df_clean['Ticket_Soporte_Original'] = df_clean['Ticket_Soporte_Abierto']
    df_clean['Ticket_Soporte_Abierto'] = df_clean['Ticket_Soporte_Abierto'].apply(
        lambda x: TICKET_MAPPING.get(str(x).lower().strip(), None) if pd.notna(x) else None
    )
    cleaning_log['acciones'].append("Normalización de Ticket_Soporte_Abierto completada")
    
    # 3. Normalizar Recomienda_Marca
    df_clean['Recomienda_Original'] = df_clean['Recomienda_Marca']
    df_clean['Recomienda_Marca'] = df_clean['Recomienda_Marca'].apply(
        lambda x: RECOMIENDA_MAPPING.get(str(x).lower().strip(), 'Sin respuesta') 
        if pd.notna(x) else 'Sin respuesta'
    )
    cleaning_log['acciones'].append("Normalización de Recomienda_Marca completada")
    
    # 4. Tratar edades imposibles - GUARDAR ORIGINALES ANTES DE CORREGIR
    df_clean['Edad_Original'] = df_clean['Edad_Cliente'].copy()
    edades_invalidas_mask = (df_clean['Edad_Cliente'] < 18) | (df_clean['Edad_Cliente'] > 100)
    df_clean['Edad_Invalida_Flag'] = edades_invalidas_mask
    
    # Guardar DataFrame original de edades inválidas ANTES de corregir
    if edades_invalidas_mask.sum() > 0:
        cleaning_log['outliers_dataframes']['edades_invalidas'] = df_clean[edades_invalidas_mask][
            ['Feedback_ID', 'Transaccion_ID', 'Edad_Original', 'Rating_Producto', 'Rating_Logistica', 'Satisfaccion_NPS']
        ].copy()
    
    # Imputar edades inválidas con la mediana
    mediana_edad = df_clean.loc[
        (df_clean['Edad_Cliente'] >= 18) & (df_clean['Edad_Cliente'] <= 100), 
        'Edad_Cliente'
    ].median()
    
    df_clean.loc[df_clean['Edad_Invalida_Flag'], 'Edad_Cliente'] = mediana_edad
    
    cleaning_log['imputaciones']['Edad_Cliente'] = {
        'metodo': 'Mediana de edades válidas (18-100 años)',
        'justificacion': 'La mediana es robusta y representa el cliente típico. Edades fuera de rango son claramente errores de captura.',
        'valores_imputados': int(edades_invalidas_mask.sum()),
        'valor_mediana': round(mediana_edad, 1)
    }
    
    # 5. Tratar Rating_Producto anómalos (debe ser 1-5) - GUARDAR ORIGINALES
    df_clean['Rating_Producto_Original'] = df_clean['Rating_Producto'].copy()
    ratings_invalidos_mask = (df_clean['Rating_Producto'] < 1) | (df_clean['Rating_Producto'] > 5)
    df_clean['Rating_Producto_Invalido_Flag'] = ratings_invalidos_mask
    
    # Guardar DataFrame original de ratings inválidos ANTES de corregir
    if ratings_invalidos_mask.sum() > 0:
        cleaning_log['outliers_dataframes']['ratings_invalidos'] = df_clean[ratings_invalidos_mask][
            ['Feedback_ID', 'Transaccion_ID', 'Rating_Producto_Original', 'Rating_Logistica', 'Comentario_Texto']
        ].copy()
    
    # Capear ratings a rango válido
    df_clean.loc[df_clean['Rating_Producto'] > 5, 'Rating_Producto'] = 5
    df_clean.loc[df_clean['Rating_Producto'] < 1, 'Rating_Producto'] = 1
    
    cleaning_log['outliers_detectados']['Rating_Producto'] = {
        'cantidad': int(ratings_invalidos_mask.sum()),
        'accion': 'Valores fuera de rango 1-5 capeados a los límites'
    }
    
    # 6. Normalizar NPS (escala -100 a 100)
    # Detectar si hay valores fuera de rango
    nps_fuera_rango = ((df_clean['Satisfaccion_NPS'] < -100) | (df_clean['Satisfaccion_NPS'] > 100)).sum()
    df_clean.loc[df_clean['Satisfaccion_NPS'] < -100, 'Satisfaccion_NPS'] = -100
    df_clean.loc[df_clean['Satisfaccion_NPS'] > 100, 'Satisfaccion_NPS'] = 100
    
    # Crear categoría NPS
    def categorize_nps(score):
        if pd.isna(score):
            return 'Sin Datos'
        if score >= 50:
            return 'Promotor'
        elif score >= 0:
            return 'Pasivo'
        else:
            return 'Detractor'
    
    df_clean['Categoria_NPS'] = df_clean['Satisfaccion_NPS'].apply(categorize_nps)
    cleaning_log['acciones'].append(f"NPS categorizado: {nps_fuera_rango} valores fuera de rango normalizados")
    
    # 7. Crear segmentos de edad
    def segment_age(age):
        if age < 25:
            return '18-24'
        elif age < 35:
            return '25-34'
        elif age < 45:
            return '35-44'
        elif age < 55:
            return '45-54'
        elif age < 65:
            return '55-64'
        else:
            return '65+'
    
    df_clean['Segmento_Edad'] = df_clean['Edad_Cliente'].apply(segment_age)
    
    cleaning_log['registros_finales'] = len(df_clean)
    
    return df_clean, cleaning_log


# =============================================================================
# FUNCIONES DE INTEGRACIÓN Y FEATURE ENGINEERING
# =============================================================================

def merge_datasets(df_inventario, df_transacciones, df_feedback):
    """
    Realiza la integración estratégica de los tres datasets.
    
    Estrategia de merge:
    1. Left join transacciones -> inventario (para identificar SKUs fantasma)
    2. Left join resultado -> feedback (para enriquecer con voz del cliente)
    
    Returns:
        tuple: (df_integrado, df_skus_fantasma, merge_stats)
    """
    merge_stats = {
        'transacciones_totales': len(df_transacciones),
        'skus_en_inventario': len(df_inventario),
        'feedback_registros': len(df_feedback)
    }
    
    # Merge 1: Transacciones + Inventario
    df_merged = df_transacciones.merge(
        df_inventario,
        on='SKU_ID',
        how='left',
        indicator='_merge_inv'
    )
    
    # Identificar SKUs fantasma
    skus_fantasma = df_merged[df_merged['_merge_inv'] == 'left_only']['SKU_ID'].unique()
    df_merged['SKU_Fantasma'] = df_merged['_merge_inv'] == 'left_only'
    
    merge_stats['skus_fantasma_unicos'] = len(skus_fantasma)
    merge_stats['transacciones_sin_inventario'] = (df_merged['_merge_inv'] == 'left_only').sum()
    merge_stats['porcentaje_ventas_fantasma'] = round(
        merge_stats['transacciones_sin_inventario'] / merge_stats['transacciones_totales'] * 100, 2
    )
    
    # Merge 2: Resultado + Feedback
    df_merged = df_merged.merge(
        df_feedback,
        on='Transaccion_ID',
        how='left',
        indicator='_merge_fb'
    )
    
    merge_stats['transacciones_con_feedback'] = (df_merged['_merge_fb'] == 'both').sum()
    merge_stats['porcentaje_con_feedback'] = round(
        merge_stats['transacciones_con_feedback'] / merge_stats['transacciones_totales'] * 100, 2
    )
    
    # Limpiar columnas de merge
    df_merged = df_merged.drop(columns=['_merge_inv', '_merge_fb'])
    
    # Crear DataFrame de SKUs fantasma para análisis
    df_skus_fantasma = df_transacciones[df_transacciones['SKU_ID'].isin(skus_fantasma)].copy()
    
    return df_merged, df_skus_fantasma, merge_stats


def create_derived_features(df):
    """
    Crea variables derivadas para análisis avanzado.
    
    Features creadas:
    1. Margen_Unitario: Precio venta - Costo unitario
    2. Margen_Total: Margen unitario * Cantidad - Costo envío
    3. Margen_Porcentaje: Margen como % del precio
    4. Brecha_Entrega: Diferencia entre tiempo real y lead time esperado
    5. Ingreso_Total: Precio venta * Cantidad
    6. Rentabilidad_Neta: Ingreso - todos los costos
    """
    df_features = df.copy()
    
    # 1. Ingresos
    df_features['Ingreso_Total'] = df_features['Precio_Venta_Final'] * df_features['Cantidad_Vendida']
    
    # 2. Margen Unitario (solo donde hay datos de inventario)
    df_features['Margen_Unitario'] = np.where(
        df_features['SKU_Fantasma'] == False,
        df_features['Precio_Venta_Final'] - df_features['Costo_Unitario_USD'].fillna(0),
        np.nan
    )
    
    # 3. Margen Total
    df_features['Costo_Total'] = (
        df_features['Costo_Unitario_USD'].fillna(0) * df_features['Cantidad_Vendida'] + 
        df_features['Costo_Envio'].fillna(0)
    )
    
    df_features['Margen_Total'] = df_features['Ingreso_Total'] - df_features['Costo_Total']
    
    # 4. Margen Porcentaje
    df_features['Margen_Porcentaje'] = np.where(
        df_features['Ingreso_Total'] > 0,
        (df_features['Margen_Total'] / df_features['Ingreso_Total']) * 100,
        0
    )
    
    # 5. Flag de Margen Negativo
    df_features['Margen_Negativo'] = df_features['Margen_Total'] < 0
    
    # 6. Brecha de Entrega (tiempo real vs esperado)
    df_features['Brecha_Entrega'] = np.where(
        df_features['Lead_Time_Dias'].notna(),
        df_features['Tiempo_Entrega_Real'] - df_features['Lead_Time_Dias'],
        np.nan
    )
    
    # 7. Categoría de rendimiento de entrega
    def categorize_delivery(brecha):
        if pd.isna(brecha):
            return 'Sin Datos'
        if brecha <= -3:
            return 'Muy Adelantado'
        elif brecha < 0:
            return 'Adelantado'
        elif brecha == 0:
            return 'A Tiempo'
        elif brecha <= 3:
            return 'Leve Retraso'
        else:
            return 'Retraso Crítico'
    
    df_features['Rendimiento_Entrega'] = df_features['Brecha_Entrega'].apply(categorize_delivery)
    
    # 8. Ratio de tickets por categoría (se calculará después de agrupar)
    
    return df_features


def generate_cleaning_report(health_before, health_after, cleaning_log, dataset_name):
    """
    Genera un reporte estructurado de limpieza para un dataset.
    """
    report = {
        'dataset': dataset_name,
        'metricas_antes': {
            'health_score': health_before['health_score'],
            'completitud': health_before['completitud_pct'],
            'unicidad': health_before['unicidad_pct'],
            'registros': health_before['total_registros'],
            'celdas_nulas': health_before['celdas_nulas'],
            'duplicados': health_before['registros_duplicados']
        },
        'metricas_despues': {
            'health_score': health_after['health_score'],
            'completitud': health_after['completitud_pct'],
            'unicidad': health_after['unicidad_pct'],
            'registros': health_after['total_registros'],
            'celdas_nulas': health_after['celdas_nulas'],
            'duplicados': health_after['registros_duplicados']
        },
        'mejora_health_score': round(health_after['health_score'] - health_before['health_score'], 2),
        'acciones_realizadas': cleaning_log['acciones'],
        'outliers_detectados': cleaning_log['outliers_detectados'],
        'imputaciones': cleaning_log['imputaciones'],
        'nulidad_por_columna_antes': health_before['nulidad_por_columna'],
        'nulidad_por_columna_despues': health_after['nulidad_por_columna']
    }
    
    return report
