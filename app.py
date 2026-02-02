"""
TechLogistics S.A. - Sistema de Soporte a la Decisión (DSS)
Dashboard Principal

Autor: Consultor Data Scientist Senior
Fecha: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import io

# Importar módulos propios
from data_cleaning import (
    calculate_health_score,
    clean_inventario,
    clean_transacciones,
    clean_feedback,
    merge_datasets,
    create_derived_features,
    generate_cleaning_report,
    generate_outlier_report
)
from utils import (
    calculate_kpis,
    format_currency,
    format_percentage,
    create_health_comparison_chart,
    create_nullity_heatmap,
    create_margin_analysis_charts,
    create_logistics_charts,
    create_customer_charts,
    create_ghost_sku_charts,
    create_stock_revision_charts,
    create_fidelity_paradox_charts,
    generate_ai_insights,
    export_cleaning_report_to_csv
)

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="TechLogistics DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# =============================================================================

@st.cache_data
def load_and_process_data(inv_path, trans_path, feed_path):
    """
    Carga y procesa todos los datasets.
    Usa cache para evitar reprocesamiento.
    """
    # Cargar datos crudos
    df_inventario_raw = pd.read_csv(inv_path)
    df_transacciones_raw = pd.read_csv(trans_path)
    df_feedback_raw = pd.read_csv(feed_path)
    
    # Calcular Health Score ANTES de limpieza
    health_inv_before = calculate_health_score(df_inventario_raw, "Inventario")
    health_trans_before = calculate_health_score(df_transacciones_raw, "Transacciones")
    health_feed_before = calculate_health_score(df_feedback_raw, "Feedback")
    
    # Limpiar datos
    df_inventario, log_inv = clean_inventario(df_inventario_raw)
    df_transacciones, log_trans = clean_transacciones(df_transacciones_raw)
    df_feedback, log_feed = clean_feedback(df_feedback_raw)
    
    # Calcular Health Score DESPUÉS de limpieza
    health_inv_after = calculate_health_score(df_inventario, "Inventario")
    health_trans_after = calculate_health_score(df_transacciones, "Transacciones")
    health_feed_after = calculate_health_score(df_feedback, "Feedback")
    
    # Generar reportes de limpieza
    report_inv = generate_cleaning_report(health_inv_before, health_inv_after, log_inv, "Inventario")
    report_trans = generate_cleaning_report(health_trans_before, health_trans_after, log_trans, "Transacciones")
    report_feed = generate_cleaning_report(health_feed_before, health_feed_after, log_feed, "Feedback")
    
    # Integrar datasets
    df_merged, df_fantasma, merge_stats = merge_datasets(df_inventario, df_transacciones, df_feedback)
    
    # Crear features derivadas
    df_final = create_derived_features(df_merged)
    
    return {
        'df_inventario_raw': df_inventario_raw,
        'df_transacciones_raw': df_transacciones_raw,
        'df_feedback_raw': df_feedback_raw,
        'df_inventario': df_inventario,
        'df_transacciones': df_transacciones,
        'df_feedback': df_feedback,
        'df_merged': df_final,
        'df_fantasma': df_fantasma,
        'merge_stats': merge_stats,
        'reports': [report_inv, report_trans, report_feed],
        'cleaning_logs': {
            'inventario': log_inv,
            'transacciones': log_trans,
            'feedback': log_feed
        },
        'health_before': {
            'Inventario': health_inv_before,
            'Transacciones': health_trans_before,
            'Feedback': health_feed_before
        },
        'health_after': {
            'Inventario': health_inv_after,
            'Transacciones': health_trans_after,
            'Feedback': health_feed_after
        }
    }


def apply_filters(df, filters):
    """
    Aplica los filtros seleccionados al DataFrame.
    """
    df_filtered = df.copy()
    
    # Filtro de fechas
    if filters.get('fecha_inicio') and filters.get('fecha_fin'):
        df_filtered = df_filtered[
            (df_filtered['Fecha_Venta'] >= pd.Timestamp(filters['fecha_inicio'])) &
            (df_filtered['Fecha_Venta'] <= pd.Timestamp(filters['fecha_fin']))
        ]
    
    # Filtro de categoría
    if filters.get('categorias') and len(filters['categorias']) > 0:
        df_filtered = df_filtered[df_filtered['Categoria'].isin(filters['categorias'])]
    
    # Filtro de bodega
    if filters.get('bodegas') and len(filters['bodegas']) > 0:
        df_filtered = df_filtered[df_filtered['Bodega_Origen'].isin(filters['bodegas'])]
    
    # Filtro de ciudad
    if filters.get('ciudades') and len(filters['ciudades']) > 0:
        df_filtered = df_filtered[df_filtered['Ciudad_Destino'].isin(filters['ciudades'])]
    
    # Filtro de canal
    if filters.get('canales') and len(filters['canales']) > 0:
        df_filtered = df_filtered[df_filtered['Canal_Venta'].isin(filters['canales'])]
    
    # Filtro de incluir SKUs fantasma
    if not filters.get('incluir_fantasma', True):
        df_filtered = df_filtered[df_filtered['SKU_Fantasma'] == False]
    
    # Filtro de excluir outliers de costo
    if filters.get('excluir_outliers', False):
        df_filtered = df_filtered[df_filtered['Costo_Outlier_Flag'] == False]
    
    return df_filtered


# =============================================================================
# BARRA LATERAL
# =============================================================================

def render_sidebar(data):
    """
    Renderiza la barra lateral con filtros y controles.
    """
    st.sidebar.markdown("## 🎛️ Panel de Control")
    st.sidebar.markdown("---")
    
    df = data['df_merged']
    
    # Información general
    st.sidebar.markdown("### 📊 Resumen de Datos")
    st.sidebar.info(f"""
    **Transacciones:** {len(df):,}
    **SKUs Únicos:** {df['SKU_ID'].nunique():,}
    **Periodo:** {df['Fecha_Venta'].min().strftime('%Y-%m-%d')} a {df['Fecha_Venta'].max().strftime('%Y-%m-%d')}
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Opciones de Análisis")
    
    filters = {}
    
    filters['incluir_fantasma'] = st.sidebar.checkbox(
        "Incluir SKUs Fantasma",
        value=True,
        help="Incluir ventas de productos no catalogados en inventario"
    )
    
    filters['excluir_outliers'] = st.sidebar.checkbox(
        "Excluir Outliers de Costo",
        value=False,
        help="Excluir productos con costos anómalos detectados por IQR"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filtros")
    
    # Filtro de fechas
    col1, col2 = st.sidebar.columns(2)
    with col1:
        filters['fecha_inicio'] = st.date_input(
            "Desde",
            value=df['Fecha_Venta'].min().date(),
            min_value=df['Fecha_Venta'].min().date(),
            max_value=df['Fecha_Venta'].max().date()
        )
    with col2:
        filters['fecha_fin'] = st.date_input(
            "Hasta",
            value=df['Fecha_Venta'].max().date(),
            min_value=df['Fecha_Venta'].min().date(),
            max_value=df['Fecha_Venta'].max().date()
        )
    
    # Filtro de categoría
    categorias_disponibles = sorted(df['Categoria'].dropna().unique().tolist())
    filters['categorias'] = st.sidebar.multiselect(
        "Categorías",
        options=categorias_disponibles,
        default=[]
    )
    
    # Filtro de bodega
    bodegas_disponibles = sorted(df['Bodega_Origen'].dropna().unique().tolist())
    filters['bodegas'] = st.sidebar.multiselect(
        "Bodegas",
        options=bodegas_disponibles,
        default=[]
    )
    
    # Filtro de ciudad
    ciudades_disponibles = sorted(df['Ciudad_Destino'].dropna().unique().tolist())
    filters['ciudades'] = st.sidebar.multiselect(
        "Ciudades",
        options=ciudades_disponibles,
        default=[]
    )
    
    # Filtro de canal
    canales_disponibles = sorted(df['Canal_Venta'].dropna().unique().tolist())
    filters['canales'] = st.sidebar.multiselect(
        "Canales de Venta",
        options=canales_disponibles,
        default=[]
    )
    
    st.sidebar.markdown("---")
    
    # Botón de refrescar
    if st.sidebar.button("🔄 Refrescar Análisis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    return filters


# =============================================================================
# TABS DE CONTENIDO
# =============================================================================

def render_auditoria_tab(data):
    """
    Renderiza la pestaña de Auditoría de Calidad.
    """
    st.markdown("## 🔬 Auditoría de Calidad de Datos")
    st.markdown("""
    Esta sección presenta el análisis de calidad de los datos **antes y después** del proceso de limpieza,
    incluyendo métricas de completitud, unicidad y validez.
    """)
    
    reports = data['reports']
    
    # Health Scores comparativos
    st.markdown("### 📈 Health Score por Dataset")
    
    cols = st.columns(3)
    for i, report in enumerate(reports):
        with cols[i]:
            delta = report['mejora_health_score']
            st.metric(
                label=f"**{report['dataset']}**",
                value=f"{report['metricas_despues']['health_score']:.1f}%",
                delta=f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            )
    
    st.markdown("---")
    
    # Gráficos de comparación
    st.markdown("### 📊 Comparación Antes vs Después")
    
    for report in reports:
        with st.expander(f"📁 {report['dataset']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de health score
                fig_health = create_health_comparison_chart(report)
                st.plotly_chart(fig_health, use_container_width=True)
            
            with col2:
                # Gráfico de nulidad
                fig_null = create_nullity_heatmap(
                    report['nulidad_por_columna_despues'],
                    report['dataset']
                )
                st.plotly_chart(fig_null, use_container_width=True)
            
            # Detalles de limpieza
            st.markdown("#### 📋 Acciones de Limpieza Realizadas")
            for accion in report['acciones_realizadas']:
                st.markdown(f"- {accion}")
            
            # Imputaciones
            if report.get('imputaciones'):
                st.markdown("#### 🔧 Decisiones de Imputación")
                for col, info in report['imputaciones'].items():
                    st.markdown(f"""
                    **{col}:**
                    - Método: {info['metodo']}
                    - Justificación: {info['justificacion']}
                    - Valores imputados: {info['valores_imputados']}
                    """)
            
            # Outliers
            if report.get('outliers_detectados'):
                st.markdown("#### ⚠️ Outliers Detectados")
                for col, info in report['outliers_detectados'].items():
                    st.warning(f"**{col}:** {info.get('cantidad', 'N/A')} outliers detectados")
    
    # ==========================================================================
    # SECCIÓN DE VISUALIZACIÓN DE OUTLIERS (VALORES ORIGINALES)
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🔍 Registros Outliers Detectados (Valores Originales)")
    st.markdown("""
    A continuación puede explorar los registros específicos que fueron identificados como outliers 
    o valores anómalos en cada dataset. **Se muestran los valores ORIGINALES antes de cualquier corrección o imputación.**
    """)
    
    # Obtener los logs de limpieza con los DataFrames originales
    cleaning_logs = data.get('cleaning_logs', {})
    
    # --- OUTLIERS DE INVENTARIO ---
    st.markdown("#### 📦 Dataset: Inventario")
    log_inv = cleaning_logs.get('inventario', {})
    outliers_inv = log_inv.get('outliers_dataframes', {})
    
    # Outliers de costo (valores originales)
    df_costo_outliers = outliers_inv.get('costo_outliers')
    if df_costo_outliers is not None and len(df_costo_outliers) > 0:
        with st.expander(f"💰 Outliers de Costo Unitario ({len(df_costo_outliers)} registros)", expanded=False):
            st.markdown("""
            **Criterio de detección:** Método IQR (Rango Intercuartílico) con multiplicador 3.
            Estos productos tienen costos unitarios significativamente fuera del rango normal.
            **Nota:** Se muestran los valores ORIGINALES sin modificar.
            """)
            st.dataframe(
                df_costo_outliers.sort_values('Costo_Unitario_USD', ascending=False),
                use_container_width=True,
                height=300
            )
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Costo Mínimo", f"${df_costo_outliers['Costo_Unitario_USD'].min():,.2f}")
            with col2:
                st.metric("Costo Máximo", f"${df_costo_outliers['Costo_Unitario_USD'].max():,.2f}")
            with col3:
                st.metric("Costo Promedio", f"${df_costo_outliers['Costo_Unitario_USD'].mean():,.2f}")
    else:
        with st.expander("💰 Outliers de Costo Unitario (0 registros)", expanded=False):
            st.success("No se detectaron outliers de costo.")
    
    # Stock negativo (valores originales)
    df_stock_negativo = outliers_inv.get('stock_negativo')
    if df_stock_negativo is not None and len(df_stock_negativo) > 0:
        with st.expander(f"📉 Registros con Stock Negativo Original ({len(df_stock_negativo)} registros)", expanded=False):
            st.markdown("""
            **Anomalía:** Estos productos tenían stock negativo en el sistema original, 
            lo cual es contablemente imposible. Fueron corregidos a 0 durante la limpieza.
            **Nota:** Se muestran los valores ORIGINALES (negativos) antes de la corrección.
            """)
            st.dataframe(
                df_stock_negativo.sort_values('Stock_Original'),
                use_container_width=True,
                height=300
            )
            # Estadísticas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stock Más Negativo", f"{df_stock_negativo['Stock_Original'].min():,.0f} unidades")
            with col2:
                st.metric("Promedio Stock Negativo", f"{df_stock_negativo['Stock_Original'].mean():,.1f} unidades")
    else:
        with st.expander("📉 Registros con Stock Negativo Original (0 registros)", expanded=False):
            st.success("No se detectaron registros con stock negativo.")
    
    # --- OUTLIERS DE TRANSACCIONES ---
    st.markdown("#### 🚚 Dataset: Transacciones")
    log_trans = cleaning_logs.get('transacciones', {})
    outliers_trans = log_trans.get('outliers_dataframes', {})
    
    # Outliers de tiempo de entrega (valores originales)
    df_tiempo_outliers = outliers_trans.get('tiempo_entrega_outliers')
    if df_tiempo_outliers is not None and len(df_tiempo_outliers) > 0:
        with st.expander(f"⏱️ Outliers de Tiempo de Entrega ({len(df_tiempo_outliers)} registros)", expanded=False):
            st.markdown("""
            **Criterio de detección:** Método IQR con multiplicador 3.
            Estos pedidos tienen tiempos de entrega extremadamente altos o inusuales.
            Valores mayores a 90 días fueron capeados durante la limpieza.
            **Nota:** Se muestran los valores ORIGINALES antes del capeo.
            """)
            st.dataframe(
                df_tiempo_outliers.sort_values('Tiempo_Entrega_Original', ascending=False),
                use_container_width=True,
                height=300
            )
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tiempo Mínimo", f"{df_tiempo_outliers['Tiempo_Entrega_Original'].min():.0f} días")
            with col2:
                st.metric("Tiempo Máximo", f"{df_tiempo_outliers['Tiempo_Entrega_Original'].max():.0f} días")
            with col3:
                st.metric("Tiempo Promedio", f"{df_tiempo_outliers['Tiempo_Entrega_Original'].mean():.1f} días")
    else:
        with st.expander("⏱️ Outliers de Tiempo de Entrega (0 registros)", expanded=False):
            st.success("No se detectaron outliers de tiempo de entrega.")
    
    # Cantidades negativas (valores originales)
    df_cant_negativas = outliers_trans.get('cantidades_negativas')
    if df_cant_negativas is not None and len(df_cant_negativas) > 0:
        with st.expander(f"🔢 Transacciones con Cantidad Negativa Original ({len(df_cant_negativas)} registros)", expanded=False):
            st.markdown("""
            **Anomalía:** Estas transacciones tenían cantidades negativas, lo cual podría indicar 
            devoluciones mal registradas o errores de digitación. Fueron convertidas a valor absoluto.
            **Nota:** Se muestran los valores ORIGINALES (negativos) antes de la corrección.
            """)
            st.dataframe(
                df_cant_negativas.sort_values('Cantidad_Original'),
                use_container_width=True,
                height=300
            )
            # Estadísticas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cantidad Más Negativa", f"{df_cant_negativas['Cantidad_Original'].min():,.0f}")
            with col2:
                st.metric("Promedio", f"{df_cant_negativas['Cantidad_Original'].mean():,.1f}")
    else:
        with st.expander("🔢 Transacciones con Cantidad Negativa Original (0 registros)", expanded=False):
            st.success("No se detectaron transacciones con cantidad negativa.")
    
    # Fechas futuras (valores originales)
    df_fechas_futuras = outliers_trans.get('fechas_futuras')
    if df_fechas_futuras is not None and len(df_fechas_futuras) > 0:
        with st.expander(f"📅 Transacciones con Fecha Futura ({len(df_fechas_futuras)} registros)", expanded=False):
            st.markdown("""
            **Anomalía:** Estas transacciones tienen fechas posteriores a la fecha actual,
            lo cual indica errores en la captura de datos o problemas de sincronización de sistemas.
            **Nota:** Se muestra tanto la fecha original como la fecha parseada.
            """)
            # Mostrar columnas relevantes
            cols_mostrar = ['Transaccion_ID', 'SKU_ID', 'Fecha_Venta_Original', 'Fecha_Venta', 'Cantidad_Vendida', 'Precio_Venta_Final', 'Canal_Venta']
            cols_disponibles = [c for c in cols_mostrar if c in df_fechas_futuras.columns]
            st.dataframe(
                df_fechas_futuras[cols_disponibles].sort_values('Fecha_Venta', ascending=False),
                use_container_width=True,
                height=300
            )
    else:
        with st.expander("📅 Transacciones con Fecha Futura (0 registros)", expanded=False):
            st.success("No se detectaron transacciones con fecha futura.")
    
    # --- OUTLIERS DE FEEDBACK ---
    st.markdown("#### 👥 Dataset: Feedback")
    log_feed = cleaning_logs.get('feedback', {})
    outliers_feed = log_feed.get('outliers_dataframes', {})
    
    # Edades inválidas (valores originales)
    df_edades_invalidas = outliers_feed.get('edades_invalidas')
    if df_edades_invalidas is not None and len(df_edades_invalidas) > 0:
        with st.expander(f"🎂 Registros con Edad Inválida Original ({len(df_edades_invalidas)} registros)", expanded=False):
            st.markdown("""
            **Criterio:** Edades fuera del rango 18-100 años fueron consideradas inválidas.
            Estos valores fueron imputados con la mediana de edades válidas.
            **Nota:** Se muestran las edades ORIGINALES antes de la imputación.
            """)
            st.dataframe(
                df_edades_invalidas.sort_values('Edad_Original', ascending=False),
                use_container_width=True,
                height=300
            )
            # Estadísticas de edades inválidas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Edad Mínima", f"{df_edades_invalidas['Edad_Original'].min():.0f} años")
            with col2:
                st.metric("Edad Máxima", f"{df_edades_invalidas['Edad_Original'].max():.0f} años")
            with col3:
                st.metric("Edad Promedio", f"{df_edades_invalidas['Edad_Original'].mean():.1f} años")
    else:
        with st.expander("🎂 Registros con Edad Inválida Original (0 registros)", expanded=False):
            st.success("No se detectaron edades inválidas.")
    
    # Ratings inválidos (valores originales)
    df_ratings_invalidos = outliers_feed.get('ratings_invalidos')
    if df_ratings_invalidos is not None and len(df_ratings_invalidos) > 0:
        with st.expander(f"⭐ Registros con Rating Inválido Original ({len(df_ratings_invalidos)} registros)", expanded=False):
            st.markdown("""
            **Criterio:** Ratings fuera del rango 1-5 fueron considerados inválidos.
            Estos valores fueron capeados a los límites del rango válido.
            **Nota:** Se muestran los ratings ORIGINALES antes del capeo.
            """)
            st.dataframe(
                df_ratings_invalidos.sort_values('Rating_Producto_Original', ascending=False),
                use_container_width=True,
                height=300
            )
            # Estadísticas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rating Mínimo", f"{df_ratings_invalidos['Rating_Producto_Original'].min():.0f}")
            with col2:
                st.metric("Rating Máximo", f"{df_ratings_invalidos['Rating_Producto_Original'].max():.0f}")
    else:
        with st.expander("⭐ Registros con Rating Inválido Original (0 registros)", expanded=False):
            st.success("No se detectaron ratings inválidos.")
    
    # Estadísticas de integración
    st.markdown("---")
    st.markdown("### 🔗 Estadísticas de Integración")
    
    merge_stats = data['merge_stats']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transacciones Totales", f"{merge_stats['transacciones_totales']:,}")
    with col2:
        st.metric("SKUs en Inventario", f"{merge_stats['skus_en_inventario']:,}")
    with col3:
        st.metric("SKUs Fantasma", f"{merge_stats['skus_fantasma_unicos']:,}")
    with col4:
        st.metric("% Ventas Sin Catálogo", f"{merge_stats['porcentaje_ventas_fantasma']:.1f}%")
    
    # Botón de descarga del reporte
    st.markdown("---")
    st.markdown("### 📥 Descargar Reporte de Limpieza")
    
    # Crear CSV del reporte
    report_buffer = io.StringIO()
    df_report = export_cleaning_report_to_csv(reports, report_buffer)
    
    # Preparar contenido para descarga
    report_content = []
    report_content.append("REPORTE DE AUDITORÍA DE CALIDAD DE DATOS")
    report_content.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append("=" * 80)
    report_content.append("")
    
    for report in reports:
        report_content.append(f"\n{'='*40}")
        report_content.append(f"DATASET: {report['dataset']}")
        report_content.append(f"{'='*40}")
        report_content.append(f"\nMÉTRICAS ANTES:")
        report_content.append(f"  - Health Score: {report['metricas_antes']['health_score']}%")
        report_content.append(f"  - Completitud: {report['metricas_antes']['completitud']}%")
        report_content.append(f"  - Unicidad: {report['metricas_antes']['unicidad']}%")
        report_content.append(f"  - Registros: {report['metricas_antes']['registros']}")
        report_content.append(f"  - Celdas Nulas: {report['metricas_antes']['celdas_nulas']}")
        report_content.append(f"  - Duplicados: {report['metricas_antes']['duplicados']}")
        
        report_content.append(f"\nMÉTRICAS DESPUÉS:")
        report_content.append(f"  - Health Score: {report['metricas_despues']['health_score']}%")
        report_content.append(f"  - Completitud: {report['metricas_despues']['completitud']}%")
        report_content.append(f"  - Unicidad: {report['metricas_despues']['unicidad']}%")
        report_content.append(f"  - Registros: {report['metricas_despues']['registros']}")
        report_content.append(f"  - Celdas Nulas: {report['metricas_despues']['celdas_nulas']}")
        report_content.append(f"  - Duplicados: {report['metricas_despues']['duplicados']}")
        
        report_content.append(f"\nMEJORA EN HEALTH SCORE: +{report['mejora_health_score']}%")
        
        report_content.append(f"\nACCIONES REALIZADAS:")
        for accion in report['acciones_realizadas']:
            report_content.append(f"  - {accion}")
        
        if report.get('imputaciones'):
            report_content.append(f"\nIMPUTACIONES:")
            for col, info in report['imputaciones'].items():
                report_content.append(f"  {col}:")
                report_content.append(f"    - Método: {info['metodo']}")
                report_content.append(f"    - Justificación: {info['justificacion']}")
                report_content.append(f"    - Valores imputados: {info['valores_imputados']}")
        
        if report.get('outliers_detectados'):
            report_content.append(f"\nOUTLIERS DETECTADOS:")
            for col, info in report['outliers_detectados'].items():
                report_content.append(f"  {col}: {info}")
        
        report_content.append(f"\nNULIDAD POR COLUMNA (DESPUÉS):")
        for col, pct in report['nulidad_por_columna_despues'].items():
            report_content.append(f"  - {col}: {pct}%")
    
    report_text = "\n".join(report_content)
    
    st.download_button(
        label="📄 Descargar Reporte Completo (TXT)",
        data=report_text,
        file_name=f"reporte_auditoria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


def render_operaciones_tab(df_filtered, data):
    """
    Renderiza la pestaña de Operaciones (Preguntas 1, 2, 3 de Alta Gerencia).
    """
    st.markdown("## 🏭 Análisis Operacional")
    
    kpis = calculate_kpis(df_filtered)
    
    # KPIs principales
    st.markdown("### 📊 KPIs Operacionales")
    
    # Usar contenedores con markdown para evitar truncamiento
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("**Ingresos Totales**")
        st.markdown(f"### ${kpis['ingresos_totales']:,.2f}")
    with col2:
        st.markdown("**Margen Total**")
        st.markdown(f"### ${kpis['margen_total']:,.2f}")
        st.caption(f"📈 {kpis['margen_porcentaje_global']:.1f}%")
    with col3:
        st.markdown("**Pérdidas (Margen -)**")
        st.markdown(f"### ${kpis['perdidas_margen_negativo']:,.2f}")
        st.caption(f"🔻 {kpis['transacciones_margen_negativo']:,} txn")
    with col4:
        st.markdown("**Tiempo Entrega Prom.**")
        st.markdown(f"### {kpis['tiempo_entrega_promedio']:.1f} días")
    with col5:
        st.markdown("**Entregas Retrasadas**")
        st.markdown(f"### {kpis['porcentaje_entregas_retrasadas']:.1f}%")
        st.caption(f"📦 {kpis['entregas_retrasadas']:,} entregas")
    
    st.markdown("---")
    
    # ==========================================================================
    # PREGUNTA 1: Fuga de Capital y Rentabilidad
    # ==========================================================================
    st.markdown("### 💰 Pregunta 1: Fuga de Capital y Rentabilidad")
    st.markdown("""
    > *¿Cuáles SKUs se venden con margen negativo? ¿Es pérdida aceptable por volumen 
    > o falla crítica de precios en el canal Online?*
    """)
    
    margin_charts = create_margin_analysis_charts(df_filtered)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(margin_charts['distribucion_margen'], use_container_width=True)
    with col2:
        st.plotly_chart(margin_charts['margen_categoria'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(margin_charts['skus_perdida'], use_container_width=True)
    with col2:
        st.plotly_chart(margin_charts['margen_canal'], use_container_width=True)
    
    # Análisis adicional por canal
    df_valid = df_filtered[df_filtered['SKU_Fantasma'] == False]
    canal_perdidas = df_valid[df_valid['Margen_Negativo'] == True].groupby('Canal_Venta').agg({
        'Margen_Total': 'sum',
        'Transaccion_ID': 'count'
    }).reset_index()
    canal_perdidas.columns = ['Canal', 'Pérdida Total', 'Transacciones']
    
    st.markdown("#### 📋 Resumen de Pérdidas por Canal")
    st.dataframe(
        canal_perdidas.style.format({'Pérdida Total': '${:,.2f}'}),
        use_container_width=True
    )
    
    # Gráfico de pérdidas por canal
    fig_perdida_canal = px.bar(
        canal_perdidas,
        x='Canal',
        y='Pérdida Total',
        color='Transacciones',
        title='Pérdidas por Margen Negativo según Canal de Venta',
        color_continuous_scale='Reds'
    )
    fig_perdida_canal.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig_perdida_canal, use_container_width=True)
    
    st.markdown("---")
    
    # ==========================================================================
    # PREGUNTA 2: Crisis Logística y Cuellos de Botella
    # ==========================================================================
    st.markdown("### 🚚 Pregunta 2: Crisis Logística y Cuellos de Botella")
    st.markdown("""
    > *¿En qué ciudades y bodegas la correlación entre Tiempo de Entrega y NPS bajo es más fuerte?
    > ¿Qué zona requiere cambio inmediato de operador?*
    """)
    
    logistics_charts = create_logistics_charts(df_filtered)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(logistics_charts['tiempo_nps_ciudad'], use_container_width=True)
    with col2:
        st.plotly_chart(logistics_charts['estado_envio'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(logistics_charts['rendimiento_bodega'], use_container_width=True)
    with col2:
        st.plotly_chart(logistics_charts['evolucion_entrega'], use_container_width=True)
    
    # Análisis de correlación ciudad-NPS
    df_logistica = df_filtered[df_filtered['Satisfaccion_NPS'].notna()].copy()
    correlacion_ciudad = df_logistica.groupby('Ciudad_Destino').agg({
        'Tiempo_Entrega_Real': 'mean',
        'Satisfaccion_NPS': 'mean',
        'Transaccion_ID': 'count'
    }).reset_index()
    correlacion_ciudad.columns = ['Ciudad', 'Tiempo Entrega', 'NPS', 'Transacciones']
    correlacion_ciudad = correlacion_ciudad[correlacion_ciudad['Transacciones'] >= 30]
    correlacion_ciudad['Riesgo'] = correlacion_ciudad['Tiempo Entrega'] / correlacion_ciudad['NPS'].abs().clip(lower=1)
    correlacion_ciudad = correlacion_ciudad.sort_values('Riesgo', ascending=False)
    
    st.markdown("#### 🎯 Ciudades con Mayor Riesgo Logístico")
    st.dataframe(
        correlacion_ciudad.head(10).style.format({
            'Tiempo Entrega': '{:.1f} días',
            'NPS': '{:.1f}',
            'Riesgo': '{:.2f}'
        }),
        use_container_width=True
    )
    
    # Matriz de correlación Tiempo de Entrega vs NPS por Ciudad
    st.markdown("#### 📊 Matriz de Correlación: Tiempo de Entrega vs NPS")
    
    # Crear datos para la matriz de correlación
    df_corr_data = df_filtered[df_filtered['Satisfaccion_NPS'].notna()].copy()
    
    # Calcular correlación por ciudad
    ciudades_con_datos = df_corr_data.groupby('Ciudad_Destino').filter(lambda x: len(x) >= 20)['Ciudad_Destino'].unique()
    
    correlaciones_ciudad = []
    for ciudad in ciudades_con_datos:
        df_ciudad = df_corr_data[df_corr_data['Ciudad_Destino'] == ciudad]
        if len(df_ciudad) >= 20:
            corr = df_ciudad['Tiempo_Entrega_Real'].corr(df_ciudad['Satisfaccion_NPS'])
            correlaciones_ciudad.append({
                'Ciudad': ciudad,
                'Correlación': corr,
                'N_Muestras': len(df_ciudad),
                'Tiempo_Prom': df_ciudad['Tiempo_Entrega_Real'].mean(),
                'NPS_Prom': df_ciudad['Satisfaccion_NPS'].mean()
            })
    
    if correlaciones_ciudad:
        df_correlaciones = pd.DataFrame(correlaciones_ciudad).sort_values('Correlación')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras de correlación por ciudad
            fig_corr_bars = px.bar(
                df_correlaciones,
                x='Correlación',
                y='Ciudad',
                orientation='h',
                color='Correlación',
                color_continuous_scale='RdYlGn_r',
                title='Correlación Tiempo-NPS por Ciudad',
                hover_data=['N_Muestras', 'Tiempo_Prom', 'NPS_Prom']
            )
            fig_corr_bars.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_corr_bars.update_layout(template='plotly_white', height=450)
            st.plotly_chart(fig_corr_bars, use_container_width=True)
        
        with col2:
            # Crear matriz de correlación con variables numéricas
            variables_corr = ['Tiempo_Entrega_Real', 'Satisfaccion_NPS', 'Rating_Logistica', 'Rating_Producto']
            df_vars = df_corr_data[variables_corr].dropna()
            
            if len(df_vars) > 0:
                matriz_corr = df_vars.corr()
                
                fig_matriz = px.imshow(
                    matriz_corr,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title='Matriz de Correlación: Variables de Servicio',
                    aspect='auto'
                )
                fig_matriz.update_layout(template='plotly_white', height=450)
                st.plotly_chart(fig_matriz, use_container_width=True)
        
        # Estadísticas de correlación
        corr_global = df_corr_data['Tiempo_Entrega_Real'].corr(df_corr_data['Satisfaccion_NPS'])
        ciudades_corr_negativa = df_correlaciones[df_correlaciones['Correlación'] < -0.1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlación Global", f"{corr_global:.3f}")
        with col2:
            st.metric("Ciudades con Correlación Negativa", f"{len(ciudades_corr_negativa)}")
        with col3:
            if len(ciudades_corr_negativa) > 0:
                peor_ciudad = ciudades_corr_negativa.iloc[0]
                st.metric("Ciudad Más Crítica", f"{peor_ciudad['Ciudad']}", 
                         delta=f"r = {peor_ciudad['Correlación']:.3f}", delta_color="inverse")
        
        # Interpretación
        if corr_global < -0.3:
            st.error(f"""
            ⚠️ **Alerta Crítica:** La correlación global es **{corr_global:.3f}**, lo que indica una relación 
            negativa moderada-fuerte entre el tiempo de entrega y la satisfacción del cliente.
            A mayor tiempo de entrega, menor NPS.
            """)
        elif corr_global < -0.1:
            st.warning(f"""
            ⚡ **Atención:** La correlación global es **{corr_global:.3f}**, indicando una relación negativa 
            leve entre tiempo de entrega y satisfacción. Se recomienda optimizar la logística.
            """)
        else:
            st.info(f"""
            ℹ️ La correlación global es **{corr_global:.3f}**. No hay una relación lineal fuerte entre 
            tiempo de entrega y NPS a nivel global, pero pueden existir patrones por ciudad o segmento.
            """)
    
    # Mapa de calor bodega-ciudad
    heatmap_data = df_filtered.groupby(['Bodega_Origen', 'Ciudad_Destino']).agg({
        'Tiempo_Entrega_Real': 'mean'
    }).reset_index()
    heatmap_data = heatmap_data[heatmap_data['Bodega_Origen'].notna()]
    heatmap_pivot = heatmap_data.pivot(index='Bodega_Origen', columns='Ciudad_Destino', values='Tiempo_Entrega_Real')
    
    fig_heatmap = px.imshow(
        heatmap_pivot,
        color_continuous_scale='RdYlGn_r',
        title='Tiempo de Entrega: Bodega → Ciudad (días)',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # ==========================================================================
    # PREGUNTA 3: Análisis de la Venta Invisible (SKUs Fantasma)
    # ==========================================================================
    st.markdown("### 👻 Pregunta 3: Análisis de la Venta Invisible")
    st.markdown("""
    > *¿Cuál es el impacto financiero de las ventas de SKUs no catalogados?
    > ¿Qué porcentaje del ingreso total está en riesgo por falta de control de inventario?*
    """)
    
    ghost_charts = create_ghost_sku_charts(df_filtered, data['df_fantasma'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(ghost_charts['comparativa_fantasma'], use_container_width=True)
    with col2:
        st.plotly_chart(ghost_charts['fantasma_canal'], use_container_width=True)
    
    st.plotly_chart(ghost_charts['impacto_fantasma'], use_container_width=True)
    
    # Resumen financiero de SKUs fantasma
    fantasma_stats = df_filtered[df_filtered['SKU_Fantasma'] == True].agg({
        'Ingreso_Total': ['sum', 'mean', 'count'],
        'Cantidad_Vendida': 'sum'
    })
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ingresos en Riesgo", format_currency(kpis['ingresos_sku_fantasma']))
    with col2:
        st.metric("% del Ingreso Total", format_percentage(kpis['porcentaje_ingresos_fantasma']))
    with col3:
        st.metric("Transacciones Afectadas", f"{kpis['ventas_sku_fantasma']:,}")
    with col4:
        skus_fantasma_unicos = df_filtered[df_filtered['SKU_Fantasma'] == True]['SKU_ID'].nunique()
        st.metric("SKUs No Catalogados", f"{skus_fantasma_unicos:,}")
    
    # Lista de SKUs fantasma
    with st.expander("📋 Ver Lista Completa de SKUs Fantasma"):
        df_fantasma_detail = df_filtered[df_filtered['SKU_Fantasma'] == True].groupby('SKU_ID').agg({
            'Ingreso_Total': 'sum',
            'Cantidad_Vendida': 'sum',
            'Transaccion_ID': 'count',
            'Canal_Venta': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
        }).reset_index().sort_values('Ingreso_Total', ascending=False)
        df_fantasma_detail.columns = ['SKU_ID', 'Ingresos', 'Unidades', 'Transacciones', 'Canal Principal']
        st.dataframe(
            df_fantasma_detail.style.format({'Ingresos': '${:,.2f}'}),
            use_container_width=True
        )


def render_cliente_tab(df_filtered, data):
    """
    Renderiza la pestaña de Cliente (Preguntas 4 y 5 de Alta Gerencia).
    """
    st.markdown("## 👥 Análisis de Cliente y Satisfacción")
    
    kpis = calculate_kpis(df_filtered)
    
    # KPIs de cliente
    st.markdown("### 📊 KPIs de Satisfacción")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("NPS Promedio", f"{kpis['nps_promedio']:.1f}",
                 delta="Promotor" if kpis['nps_promedio'] >= 50 else ("Pasivo" if kpis['nps_promedio'] >= 0 else "Detractor"))
    with col2:
        st.metric("Rating Producto", f"{kpis['rating_producto_promedio']:.2f}/5")
    with col3:
        st.metric("Rating Logística", f"{kpis['rating_logistica_promedio']:.2f}/5")
    with col4:
        st.metric("Tasa Tickets Soporte", format_percentage(kpis['porcentaje_tickets_soporte']))
    
    st.markdown("---")
    
    # ==========================================================================
    # PREGUNTA 4: Diagnóstico de Fidelidad (Paradoja Stock-Satisfacción)
    # ==========================================================================
    st.markdown("### 🎭 Pregunta 4: Diagnóstico de Fidelidad")
    st.markdown("""
    > *¿Existen categorías con alta disponibilidad (stock alto) pero sentimiento negativo?
    > ¿Es mala calidad de producto o sobrecosto?*
    """)
    
    paradox_charts = create_fidelity_paradox_charts(df_filtered, data['df_inventario'])
    customer_charts = create_customer_charts(df_filtered)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(paradox_charts['paradoja_stock_nps'], use_container_width=True)
    with col2:
        if 'nps_categoria' in customer_charts:
            st.plotly_chart(customer_charts['nps_categoria'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(paradox_charts['matriz_rendimiento'], use_container_width=True)
    with col2:
        if 'tickets_categoria' in customer_charts:
            st.plotly_chart(customer_charts['tickets_categoria'], use_container_width=True)
    
    # Análisis de la paradoja
    st.markdown("#### 🔍 Análisis de la Paradoja Stock-Satisfacción")
    
    # Calcular métricas por categoría
    df_fb = df_filtered[df_filtered['Rating_Producto'].notna()].copy()
    stock_cat = data['df_inventario'].groupby('Categoria')['Stock_Actual'].sum().reset_index()
    stock_cat.columns = ['Categoria', 'Stock_Total']
    
    sentiment_cat = df_fb.groupby('Categoria').agg({
        'Satisfaccion_NPS': 'mean',
        'Rating_Producto': 'mean',
        'Precio_Venta_Final': 'mean',
        'Costo_Unitario_USD': 'mean'
    }).reset_index()
    sentiment_cat.columns = ['Categoria', 'NPS', 'Rating', 'Precio_Prom', 'Costo_Prom']
    
    paradox_analysis = stock_cat.merge(sentiment_cat, on='Categoria', how='inner')
    paradox_analysis = paradox_analysis[paradox_analysis['Categoria'].notna()]
    paradox_analysis['Margen_Prom'] = paradox_analysis['Precio_Prom'] - paradox_analysis['Costo_Prom']
    paradox_analysis['Stock_Normalizado'] = paradox_analysis['Stock_Total'] / paradox_analysis['Stock_Total'].max() * 100
    
    # Identificar categorías con paradoja
    alto_stock = paradox_analysis['Stock_Normalizado'] > 50
    bajo_nps = paradox_analysis['NPS'] < paradox_analysis['NPS'].median()
    paradox_analysis['Paradoja'] = alto_stock & bajo_nps
    
    categorias_paradoja = paradox_analysis[paradox_analysis['Paradoja'] == True]
    
    if len(categorias_paradoja) > 0:
        st.warning(f"⚠️ Se detectaron {len(categorias_paradoja)} categorías con la paradoja Alto Stock + Bajo NPS")
        st.dataframe(
            categorias_paradoja[['Categoria', 'Stock_Total', 'NPS', 'Rating', 'Precio_Prom', 'Costo_Prom', 'Margen_Prom']].style.format({
                'Stock_Total': '{:,.0f}',
                'NPS': '{:.1f}',
                'Rating': '{:.2f}',
                'Precio_Prom': '${:,.2f}',
                'Costo_Prom': '${:,.2f}',
                'Margen_Prom': '${:,.2f}'
            }),
            use_container_width=True
        )
    else:
        st.success("✅ No se detectaron categorías con la paradoja Alto Stock + Bajo NPS")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(customer_charts['rating_scatter'], use_container_width=True)
    with col2:
        if 'recomendacion' in customer_charts:
            st.plotly_chart(customer_charts['recomendacion'], use_container_width=True)
    
    st.markdown("---")
    
    # ==========================================================================
    # PREGUNTA 5: Storytelling de Riesgo Operativo
    # ==========================================================================
    st.markdown("### 🏚️ Pregunta 5: Storytelling de Riesgo Operativo")
    st.markdown("""
    > *¿Qué relación existe entre la antigüedad de la última revisión de stock y la tasa de tickets de soporte?
    > ¿Qué bodegas operan a ciegas?*
    """)
    
    revision_charts = create_stock_revision_charts(df_filtered, data['df_inventario'])
    
    col1, col2 = st.columns(2)
    with col1:
        if 'dias_revision_bodega' in revision_charts:
            st.plotly_chart(revision_charts['dias_revision_bodega'], use_container_width=True)
    with col2:
        if 'tickets_vs_revision' in revision_charts:
            st.plotly_chart(revision_charts['tickets_vs_revision'], use_container_width=True)
    
    if 'bodegas_ciegas' in revision_charts:
        st.plotly_chart(revision_charts['bodegas_ciegas'], use_container_width=True)
    
    # Análisis de bodegas operando a ciegas
    st.markdown("#### 🔦 Bodegas Operando a Ciegas (Sin Revisión >180 días)")
    
    df_inv = data['df_inventario']
    bodegas_ciegas = df_inv[df_inv['Dias_Sin_Revision'] > 180].groupby('Bodega_Origen').agg({
        'SKU_ID': 'count',
        'Stock_Actual': 'sum',
        'Dias_Sin_Revision': ['mean', 'max']
    }).reset_index()
    bodegas_ciegas.columns = ['Bodega', 'SKUs Desactualizados', 'Stock en Riesgo', 'Días Promedio', 'Días Máximo']
    
    if len(bodegas_ciegas) > 0:
        st.error(f"🚨 {len(bodegas_ciegas)} bodegas tienen productos sin revisión por más de 180 días")
        st.dataframe(
            bodegas_ciegas.style.format({
                'Stock en Riesgo': '{:,.0f}',
                'Días Promedio': '{:.0f}',
                'Días Máximo': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Impacto en tickets
        df_merge_revision = df_filtered[df_filtered['Dias_Sin_Revision'].notna() & df_filtered['Ticket_Soporte_Abierto'].notna()].copy()
        if len(df_merge_revision) > 0:
            corr_revision_tickets = df_merge_revision['Dias_Sin_Revision'].corr(
                df_merge_revision['Ticket_Soporte_Abierto'].astype(float)
            )
            st.info(f"📈 Correlación entre días sin revisión y tickets de soporte: **{corr_revision_tickets:.3f}**")
    else:
        st.success("✅ Todas las bodegas tienen revisiones recientes (<180 días)")


def render_insights_tab(df_filtered, data):
    """
    Renderiza la pestaña de Insights de IA.
    """
    st.markdown("## 🤖 Insights Generados por IA")
    st.markdown("""
    Esta sección utiliza el modelo **Llama-3** de Groq para generar recomendaciones estratégicas
    basadas en los datos filtrados actualmente.
    """)
    
    # Input de API Key
    st.markdown("### 🔑 Configuración de API")
    
    # Intentar obtener API key de secrets o variable de entorno
    api_key = st.text_input(
        "API Key de Groq",
        type="password",
        help="Ingrese su API Key de Groq. Obténgala en https://console.groq.com/",
        value=os.environ.get('GROQ_API_KEY', '')
    )
    
    if not api_key:
        st.warning("⚠️ Ingrese su API Key de Groq para generar insights con IA")
        st.info("""
        **¿No tienes una API Key?**
        1. Ve a [console.groq.com](https://console.groq.com/)
        2. Crea una cuenta gratuita
        3. Genera una API Key en la sección 'API Keys'
        """)
        return
    
    # Mostrar resumen de datos actuales
    st.markdown("### 📊 Datos Actualmente Filtrados")
    
    kpis = calculate_kpis(df_filtered)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transacciones", f"{len(df_filtered):,}")
    with col2:
        st.metric("Ingresos", format_currency(kpis['ingresos_totales']))
    with col3:
        st.metric("Margen", format_percentage(kpis['margen_porcentaje_global']))
    
    st.markdown("---")
    
    # Botón para generar insights
    if st.button("🚀 Generar Recomendaciones Estratégicas", use_container_width=True, type="primary"):
        with st.spinner("Analizando datos y generando recomendaciones..."):
            try:
                insights = generate_ai_insights(df_filtered, kpis, api_key)
                
                st.markdown("### 💡 Recomendaciones Estratégicas")
                st.markdown("---")
                st.markdown(insights)
                st.markdown("---")
                
                # Opción de descarga
                st.download_button(
                    label="📥 Descargar Recomendaciones",
                    data=f"RECOMENDACIONES ESTRATÉGICAS - TechLogistics S.A.\n\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{insights}",
                    file_name=f"recomendaciones_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error al generar insights: {str(e)}")
                st.info("Verifique que su API Key sea válida y tenga créditos disponibles.")


# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal de la aplicación.
    """
    # Header
    st.markdown('<p class="main-header">🏢 TechLogistics S.A.</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dashboard | Sistema de Soporte a la Decisión (DSS)</p>', unsafe_allow_html=True)
    
    # Cargar datos
    try:
        # Rutas de los archivos
        data_path = "datasets"
        inv_path = os.path.join(data_path, "inventario_central_v2.csv")
        trans_path = os.path.join(data_path, "transacciones_logistica_v2.csv")
        feed_path = os.path.join(data_path, "feedback_clientes_v2.csv")
        
        # Verificar existencia de archivos
        if not all(os.path.exists(p) for p in [inv_path, trans_path, feed_path]):
            st.error("⚠️ No se encontraron los archivos de datos. Asegúrese de que los archivos CSV estén en la carpeta 'datasets/'")
            st.info("""
            Archivos requeridos:
            - datasets/inventario_central_v2.csv
            - datasets/transacciones_logistica_v2.csv
            - datasets/feedback_clientes_v2.csv
            """)
            return
        
        # Cargar y procesar datos
        with st.spinner("Cargando y procesando datos..."):
            data = load_and_process_data(inv_path, trans_path, feed_path)
        
        # Renderizar sidebar y obtener filtros
        filters = render_sidebar(data)
        
        # Aplicar filtros
        df_filtered = apply_filters(data['df_merged'], filters)
        
        # Mostrar contador de registros filtrados
        st.markdown(f"**📋 Registros mostrados:** {len(df_filtered):,} de {len(data['df_merged']):,}")
        
        # Tabs principales
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Auditoría de Datos",
            "🏭 Operaciones",
            "👥 Cliente",
            "🤖 Insights IA"
        ])
        
        with tab1:
            render_auditoria_tab(data)
        
        with tab2:
            render_operaciones_tab(df_filtered, data)
        
        with tab3:
            render_cliente_tab(df_filtered, data)
        
        with tab4:
            render_insights_tab(df_filtered, data)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        TechLogistics S.A. DSS | Desarrollado por Gia Mariana Calle Higuita - José Santiago Molano Perdomo - Juan José Restrepo Higuita | SI6001 - Fundamentos en Ciencias de Datos | 2026
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error al cargar la aplicación: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
