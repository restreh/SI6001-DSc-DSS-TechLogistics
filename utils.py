"""
Módulo de Utilidades
TechLogistics S.A. - Sistema de Soporte a la Decisión (DSS)

Este módulo contiene funciones auxiliares para:
- Generación de visualizaciones
- Cálculo de KPIs
- Funciones de formato y presentación
- Integración con Groq API
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime


# =============================================================================
# FUNCIONES DE KPIs
# =============================================================================

def calculate_kpis(df):
    """
    Calcula los KPIs principales del negocio.
    """
    kpis = {}
    
    # Filtrar solo datos válidos (sin SKUs fantasma para métricas de margen)
    df_valid = df[df['SKU_Fantasma'] == False].copy()
    
    # Ingresos
    kpis['ingresos_totales'] = df['Ingreso_Total'].sum()
    kpis['ingresos_promedio_transaccion'] = df['Ingreso_Total'].mean()
    
    # Márgenes (solo con datos válidos)
    kpis['margen_total'] = df_valid['Margen_Total'].sum()
    kpis['margen_promedio'] = df_valid['Margen_Total'].mean()
    kpis['margen_porcentaje_global'] = (kpis['margen_total'] / df_valid['Ingreso_Total'].sum() * 100) if df_valid['Ingreso_Total'].sum() > 0 else 0
    
    # Pérdidas por margen negativo
    df_perdidas = df_valid[df_valid['Margen_Negativo'] == True]
    kpis['perdidas_margen_negativo'] = abs(df_perdidas['Margen_Total'].sum())
    kpis['transacciones_margen_negativo'] = len(df_perdidas)
    kpis['porcentaje_transacciones_perdida'] = len(df_perdidas) / len(df_valid) * 100 if len(df_valid) > 0 else 0
    
    # Logística
    kpis['tiempo_entrega_promedio'] = df['Tiempo_Entrega_Real'].mean()
    kpis['entregas_retrasadas'] = (df['Rendimiento_Entrega'].isin(['Leve Retraso', 'Retraso Crítico'])).sum()
    kpis['porcentaje_entregas_retrasadas'] = kpis['entregas_retrasadas'] / len(df) * 100
    
    # SKUs Fantasma
    kpis['ventas_sku_fantasma'] = (df['SKU_Fantasma'] == True).sum()
    kpis['ingresos_sku_fantasma'] = df[df['SKU_Fantasma'] == True]['Ingreso_Total'].sum()
    kpis['porcentaje_ingresos_fantasma'] = kpis['ingresos_sku_fantasma'] / kpis['ingresos_totales'] * 100 if kpis['ingresos_totales'] > 0 else 0
    
    # Satisfacción del cliente (donde hay feedback)
    df_feedback = df[df['Rating_Producto'].notna()]
    if len(df_feedback) > 0:
        kpis['nps_promedio'] = df_feedback['Satisfaccion_NPS'].mean()
        kpis['rating_producto_promedio'] = df_feedback['Rating_Producto'].mean()
        kpis['rating_logistica_promedio'] = df_feedback['Rating_Logistica'].mean()
        kpis['porcentaje_tickets_soporte'] = df_feedback['Ticket_Soporte_Abierto'].sum() / len(df_feedback) * 100 if 'Ticket_Soporte_Abierto' in df_feedback.columns else 0
    else:
        kpis['nps_promedio'] = 0
        kpis['rating_producto_promedio'] = 0
        kpis['rating_logistica_promedio'] = 0
        kpis['porcentaje_tickets_soporte'] = 0
    
    return kpis


def format_currency(value, prefix='$', decimals=2):
    """Formatea un valor como moneda."""
    if pd.isna(value):
        return f"{prefix}0.00"
    return f"{prefix}{value:,.{decimals}f}"


def format_percentage(value, decimals=1):
    """Formatea un valor como porcentaje."""
    if pd.isna(value):
        return "0.0%"
    return f"{value:.{decimals}f}%"


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def create_health_comparison_chart(report):
    """
    Crea un gráfico comparativo del Health Score antes/después.
    """
    categories = ['Health Score', 'Completitud', 'Unicidad']
    before = [
        report['metricas_antes']['health_score'],
        report['metricas_antes']['completitud'],
        report['metricas_antes']['unicidad']
    ]
    after = [
        report['metricas_despues']['health_score'],
        report['metricas_despues']['completitud'],
        report['metricas_despues']['unicidad']
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Antes', x=categories, y=before, marker_color='#EF553B'),
        go.Bar(name='Después', x=categories, y=after, marker_color='#00CC96')
    ])
    
    fig.update_layout(
        title=f"Métricas de Calidad - {report['dataset']}",
        barmode='group',
        yaxis_title='Porcentaje (%)',
        yaxis_range=[0, 100],
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_nullity_heatmap(null_pct_dict, dataset_name):
    """
    Crea un heatmap de nulidad por columna.
    """
    columns = list(null_pct_dict.keys())
    values = list(null_pct_dict.values())
    
    fig = go.Figure(data=go.Bar(
        x=columns,
        y=values,
        marker_color=['#EF553B' if v > 5 else '#FFA15A' if v > 0 else '#00CC96' for v in values],
        text=[f'{v:.1f}%' for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Porcentaje de Nulidad por Columna - {dataset_name}',
        yaxis_title='% Nulos',
        xaxis_tickangle=-45,
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_margin_analysis_charts(df):
    """
    Crea visualizaciones para análisis de margen y rentabilidad.
    """
    df_valid = df[df['SKU_Fantasma'] == False].copy()
    
    charts = {}
    
    # 1. Distribución de márgenes
    fig_dist = px.histogram(
        df_valid,
        x='Margen_Total',
        nbins=50,
        color_discrete_sequence=['#636EFA'],
        title='Distribución del Margen por Transacción'
    )
    fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Punto de Equilibrio")
    fig_dist.update_layout(template='plotly_white', height=400)
    charts['distribucion_margen'] = fig_dist
    
    # 2. Margen por categoría
    margin_by_cat = df_valid.groupby('Categoria').agg({
        'Margen_Total': 'sum',
        'Ingreso_Total': 'sum',
        'Transaccion_ID': 'count'
    }).reset_index()
    margin_by_cat.columns = ['Categoría', 'Margen Total', 'Ingresos', 'Transacciones']
    margin_by_cat['Margen %'] = margin_by_cat['Margen Total'] / margin_by_cat['Ingresos'] * 100
    
    fig_cat = px.bar(
        margin_by_cat.sort_values('Margen Total'),
        x='Margen Total',
        y='Categoría',
        orientation='h',
        color='Margen %',
        color_continuous_scale='RdYlGn',
        title='Margen Total por Categoría'
    )
    fig_cat.update_layout(template='plotly_white', height=400)
    charts['margen_categoria'] = fig_cat
    
    # 3. SKUs con margen negativo
    skus_perdida = df_valid[df_valid['Margen_Negativo'] == True].groupby('SKU_ID').agg({
        'Margen_Total': 'sum',
        'Cantidad_Vendida': 'sum',
        'Categoria': 'first'
    }).reset_index().sort_values('Margen_Total').head(20)
    
    fig_perdida = px.bar(
        skus_perdida,
        x='Margen_Total',
        y='SKU_ID',
        orientation='h',
        color='Categoria',
        title='Top 20 SKUs con Mayor Pérdida (Margen Negativo)'
    )
    fig_perdida.update_layout(template='plotly_white', height=500)
    charts['skus_perdida'] = fig_perdida
    
    # 4. Análisis por canal de venta
    margin_by_channel = df_valid.groupby('Canal_Venta').agg({
        'Margen_Total': ['sum', 'mean'],
        'Ingreso_Total': 'sum',
        'Transaccion_ID': 'count'
    }).reset_index()
    margin_by_channel.columns = ['Canal', 'Margen Total', 'Margen Promedio', 'Ingresos', 'Transacciones']
    margin_by_channel['Margen %'] = margin_by_channel['Margen Total'] / margin_by_channel['Ingresos'] * 100
    
    fig_channel = px.bar(
        margin_by_channel,
        x='Canal',
        y=['Margen Total', 'Ingresos'],
        barmode='group',
        title='Margen e Ingresos por Canal de Venta'
    )
    fig_channel.update_layout(template='plotly_white', height=400)
    charts['margen_canal'] = fig_channel
    
    return charts


def create_logistics_charts(df):
    """
    Crea visualizaciones para análisis logístico.
    """
    charts = {}
    
    # 1. Tiempo de entrega por ciudad
    tiempo_ciudad = df.groupby('Ciudad_Destino').agg({
        'Tiempo_Entrega_Real': 'mean',
        'Satisfaccion_NPS': 'mean',
        'Transaccion_ID': 'count'
    }).reset_index()
    tiempo_ciudad.columns = ['Ciudad', 'Tiempo Promedio', 'NPS Promedio', 'Transacciones']
    tiempo_ciudad = tiempo_ciudad[tiempo_ciudad['Transacciones'] >= 50]  # Filtrar ciudades con suficientes datos
    
    fig_ciudad = px.scatter(
        tiempo_ciudad,
        x='Tiempo Promedio',
        y='NPS Promedio',
        size='Transacciones',
        color='Ciudad',
        title='Correlación: Tiempo de Entrega vs NPS por Ciudad',
        hover_data=['Transacciones']
    )
    fig_ciudad.update_layout(template='plotly_white', height=500)
    charts['tiempo_nps_ciudad'] = fig_ciudad
    
    # 2. Estado de envíos
    estado_counts = df['Estado_Envio'].value_counts().reset_index()
    estado_counts.columns = ['Estado', 'Cantidad']
    
    fig_estado = px.pie(
        estado_counts,
        values='Cantidad',
        names='Estado',
        title='Distribución de Estados de Envío',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_estado.update_layout(template='plotly_white', height=400)
    charts['estado_envio'] = fig_estado
    
    # 3. Rendimiento de entrega por bodega
    rendimiento_bodega = df.groupby(['Bodega_Origen', 'Rendimiento_Entrega']).size().reset_index(name='Cantidad')
    rendimiento_bodega = rendimiento_bodega[rendimiento_bodega['Bodega_Origen'].notna()]
    
    fig_bodega = px.bar(
        rendimiento_bodega,
        x='Bodega_Origen',
        y='Cantidad',
        color='Rendimiento_Entrega',
        title='Rendimiento de Entrega por Bodega de Origen',
        color_discrete_map={
            'Muy Adelantado': '#00CC96',
            'Adelantado': '#19D3F3',
            'A Tiempo': '#636EFA',
            'Leve Retraso': '#FFA15A',
            'Retraso Crítico': '#EF553B',
            'Sin Datos': '#7F7F7F'
        }
    )
    fig_bodega.update_layout(template='plotly_white', height=400, barmode='stack')
    charts['rendimiento_bodega'] = fig_bodega
    
    # 4. Evolución temporal del tiempo de entrega
    df_tiempo = df.copy()
    df_tiempo['Mes'] = df_tiempo['Fecha_Venta'].dt.to_period('M').astype(str)
    tiempo_mensual = df_tiempo.groupby('Mes').agg({
        'Tiempo_Entrega_Real': 'mean',
        'Brecha_Entrega': 'mean'
    }).reset_index()
    
    fig_temporal = go.Figure()
    fig_temporal.add_trace(go.Scatter(
        x=tiempo_mensual['Mes'],
        y=tiempo_mensual['Tiempo_Entrega_Real'],
        mode='lines+markers',
        name='Tiempo Real'
    ))
    fig_temporal.add_trace(go.Scatter(
        x=tiempo_mensual['Mes'],
        y=tiempo_mensual['Brecha_Entrega'],
        mode='lines+markers',
        name='Brecha vs Lead Time'
    ))
    fig_temporal.update_layout(
        title='Evolución del Tiempo de Entrega',
        template='plotly_white',
        height=400
    )
    charts['evolucion_entrega'] = fig_temporal
    
    return charts


def create_customer_charts(df):
    """
    Crea visualizaciones para análisis de cliente.
    """
    df_fb = df[df['Rating_Producto'].notna()].copy()
    charts = {}
    
    # 1. Distribución NPS por categoría
    if len(df_fb) > 0 and 'Categoria' in df_fb.columns:
        nps_cat = df_fb.groupby('Categoria').agg({
            'Satisfaccion_NPS': 'mean',
            'Rating_Producto': 'mean',
            'Rating_Logistica': 'mean',
            'Feedback_ID': 'count'
        }).reset_index()
        nps_cat.columns = ['Categoría', 'NPS', 'Rating Producto', 'Rating Logística', 'Respuestas']
        nps_cat = nps_cat[nps_cat['Categoría'].notna()]
        
        fig_nps_cat = px.bar(
            nps_cat.sort_values('NPS'),
            x='NPS',
            y='Categoría',
            orientation='h',
            color='NPS',
            color_continuous_scale='RdYlGn',
            title='NPS Promedio por Categoría de Producto'
        )
        fig_nps_cat.update_layout(template='plotly_white', height=400)
        charts['nps_categoria'] = fig_nps_cat
    
    # 2. Relación Rating Producto vs Logística - HEATMAP DE FRECUENCIAS
    # Crear tabla de frecuencias
    rating_crosstab = pd.crosstab(df_fb['Rating_Logistica'], df_fb['Rating_Producto'])
    
    fig_heatmap_ratings = px.imshow(
        rating_crosstab,
        labels=dict(x="Rating Producto", y="Rating Logística", color="Frecuencia"),
        x=[1, 2, 3, 4, 5],
        y=[1, 2, 3, 4, 5],
        color_continuous_scale='Blues',
        title='Distribución de Ratings: Producto vs Logística',
        text_auto=True,
        aspect='equal'
    )
    fig_heatmap_ratings.update_layout(template='plotly_white', height=450)
    fig_heatmap_ratings.update_xaxes(side='bottom')
    charts['rating_scatter'] = fig_heatmap_ratings
    
    # 3. Tickets de soporte por categoría
    if 'Ticket_Soporte_Abierto' in df_fb.columns:
        tickets_cat = df_fb.groupby('Categoria').agg({
            'Ticket_Soporte_Abierto': lambda x: x.sum() if x.dtype == bool else (x == True).sum(),
            'Feedback_ID': 'count'
        }).reset_index()
        tickets_cat.columns = ['Categoría', 'Tickets', 'Total Feedback']
        tickets_cat['Tasa Tickets (%)'] = tickets_cat['Tickets'] / tickets_cat['Total Feedback'] * 100
        tickets_cat = tickets_cat[tickets_cat['Categoría'].notna()]
        
        fig_tickets = px.bar(
            tickets_cat.sort_values('Tasa Tickets (%)'),
            x='Tasa Tickets (%)',
            y='Categoría',
            orientation='h',
            color='Tasa Tickets (%)',
            color_continuous_scale='Reds',
            title='Tasa de Tickets de Soporte por Categoría'
        )
        fig_tickets.update_layout(template='plotly_white', height=400)
        charts['tickets_categoria'] = fig_tickets
    
    # 4. Distribución de recomendación
    if 'Recomienda_Marca' in df_fb.columns:
        recomienda_dist = df_fb['Recomienda_Marca'].value_counts().reset_index()
        recomienda_dist.columns = ['Respuesta', 'Cantidad']
        
        fig_rec = px.pie(
            recomienda_dist,
            values='Cantidad',
            names='Respuesta',
            title='¿Recomendaría la Marca?',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_rec.update_layout(template='plotly_white', height=400)
        charts['recomendacion'] = fig_rec
    
    return charts


def create_ghost_sku_charts(df, df_fantasma):
    """
    Crea visualizaciones para análisis de SKUs fantasma.
    """
    charts = {}
    
    # 1. Impacto financiero de SKUs fantasma
    fantasma_summary = df[df['SKU_Fantasma'] == True].groupby('SKU_ID').agg({
        'Ingreso_Total': 'sum',
        'Cantidad_Vendida': 'sum',
        'Transaccion_ID': 'count'
    }).reset_index().sort_values('Ingreso_Total', ascending=False).head(15)
    fantasma_summary.columns = ['SKU_ID', 'Ingresos', 'Unidades', 'Transacciones']
    
    fig_impacto = px.bar(
        fantasma_summary,
        x='SKU_ID',
        y='Ingresos',
        color='Transacciones',
        title='Top 15 SKUs Fantasma por Ingresos en Riesgo',
        color_continuous_scale='Reds'
    )
    fig_impacto.update_layout(template='plotly_white', height=400, xaxis_tickangle=-45)
    charts['impacto_fantasma'] = fig_impacto
    
    # 2. Distribución por canal
    fantasma_canal = df[df['SKU_Fantasma'] == True].groupby('Canal_Venta').agg({
        'Ingreso_Total': 'sum',
        'Transaccion_ID': 'count'
    }).reset_index()
    fantasma_canal.columns = ['Canal', 'Ingresos', 'Transacciones']
    
    fig_canal = px.pie(
        fantasma_canal,
        values='Ingresos',
        names='Canal',
        title='Distribución de Ventas Fantasma por Canal'
    )
    fig_canal.update_layout(template='plotly_white', height=400)
    charts['fantasma_canal'] = fig_canal
    
    # 3. Comparativa SKUs válidos vs fantasma
    comparativa = pd.DataFrame({
        'Tipo': ['SKUs Catalogados', 'SKUs Fantasma'],
        'Ingresos': [
            df[df['SKU_Fantasma'] == False]['Ingreso_Total'].sum(),
            df[df['SKU_Fantasma'] == True]['Ingreso_Total'].sum()
        ],
        'Transacciones': [
            len(df[df['SKU_Fantasma'] == False]),
            len(df[df['SKU_Fantasma'] == True])
        ]
    })
    
    fig_comp = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]],
                            subplot_titles=['Por Ingresos', 'Por Transacciones'])
    
    fig_comp.add_trace(go.Pie(
        labels=comparativa['Tipo'],
        values=comparativa['Ingresos'],
        marker_colors=['#00CC96', '#EF553B']
    ), row=1, col=1)
    
    fig_comp.add_trace(go.Pie(
        labels=comparativa['Tipo'],
        values=comparativa['Transacciones'],
        marker_colors=['#00CC96', '#EF553B']
    ), row=1, col=2)
    
    fig_comp.update_layout(title='Impacto de SKUs Fantasma', height=400, template='plotly_white')
    charts['comparativa_fantasma'] = fig_comp
    
    return charts


def create_stock_revision_charts(df, df_inventario):
    """
    Crea visualizaciones para análisis de revisión de stock y tickets.
    """
    charts = {}
    
    # 1. Días sin revisión por bodega
    if 'Dias_Sin_Revision' in df_inventario.columns:
        revision_bodega = df_inventario.groupby('Bodega_Origen').agg({
            'Dias_Sin_Revision': 'mean',
            'SKU_ID': 'count'
        }).reset_index()
        revision_bodega.columns = ['Bodega', 'Días Promedio Sin Revisión', 'SKUs']
        
        fig_revision = px.bar(
            revision_bodega.sort_values('Días Promedio Sin Revisión'),
            x='Bodega',
            y='Días Promedio Sin Revisión',
            color='Días Promedio Sin Revisión',
            color_continuous_scale='Reds',
            title='Días Promedio Sin Revisión de Stock por Bodega'
        )
        fig_revision.update_layout(template='plotly_white', height=400)
        charts['dias_revision_bodega'] = fig_revision
    
    # 2. Correlación entre antigüedad de revisión y tickets
    df_merged = df[df['Rating_Producto'].notna() & df['Dias_Sin_Revision'].notna()].copy()
    if len(df_merged) > 0 and 'Ticket_Soporte_Abierto' in df_merged.columns:
        # Agrupar por rangos de días sin revisión
        bins = [0, 30, 90, 180, 365, float('inf')]
        labels = ['0-30 días', '31-90 días', '91-180 días', '181-365 días', '+365 días']
        df_merged['Rango_Revision'] = pd.cut(df_merged['Dias_Sin_Revision'], bins=bins, labels=labels)
        
        tickets_revision = df_merged.groupby('Rango_Revision', observed=True).agg({
            'Ticket_Soporte_Abierto': lambda x: x.sum() if x.dtype == bool else (x == True).sum(),
            'Feedback_ID': 'count'
        }).reset_index()
        tickets_revision.columns = ['Rango Revisión', 'Tickets', 'Total']
        tickets_revision['Tasa Tickets (%)'] = tickets_revision['Tickets'] / tickets_revision['Total'] * 100
        
        fig_corr = px.bar(
            tickets_revision,
            x='Rango Revisión',
            y='Tasa Tickets (%)',
            color='Tasa Tickets (%)',
            color_continuous_scale='Reds',
            title='Tasa de Tickets de Soporte vs Antigüedad de Revisión de Stock'
        )
        fig_corr.update_layout(template='plotly_white', height=400)
        charts['tickets_vs_revision'] = fig_corr
    
    # 3. Bodegas operando "a ciegas"
    if 'Dias_Sin_Revision' in df_inventario.columns:
        bodegas_ciegas = df_inventario[df_inventario['Dias_Sin_Revision'] > 180].groupby('Bodega_Origen').agg({
            'SKU_ID': 'count',
            'Stock_Actual': 'sum',
            'Dias_Sin_Revision': 'mean'
        }).reset_index()
        bodegas_ciegas.columns = ['Bodega', 'SKUs Sin Revisión', 'Stock Total', 'Días Promedio']
        
        fig_ciegas = px.scatter(
            bodegas_ciegas,
            x='Días Promedio',
            y='Stock Total',
            size='SKUs Sin Revisión',
            color='Bodega',
            title='Bodegas con Stock Sin Revisión (+180 días)',
            hover_data=['SKUs Sin Revisión']
        )
        fig_ciegas.update_layout(template='plotly_white', height=400)
        charts['bodegas_ciegas'] = fig_ciegas
    
    return charts


def create_fidelity_paradox_charts(df, df_inventario):
    """
    Crea visualizaciones para el análisis de la paradoja de fidelidad.
    """
    charts = {}
    
    # Combinar datos de stock con feedback
    stock_by_cat = df_inventario.groupby('Categoria').agg({
        'Stock_Actual': 'sum',
        'SKU_ID': 'count'
    }).reset_index()
    stock_by_cat.columns = ['Categoria', 'Stock_Total', 'SKUs']
    
    df_fb = df[df['Rating_Producto'].notna()].copy()
    sentiment_by_cat = df_fb.groupby('Categoria').agg({
        'Rating_Producto': 'mean',
        'Satisfaccion_NPS': 'mean',
        'Feedback_ID': 'count'
    }).reset_index()
    sentiment_by_cat.columns = ['Categoria', 'Rating_Promedio', 'NPS_Promedio', 'Feedback_Count']
    
    paradox_df = stock_by_cat.merge(sentiment_by_cat, on='Categoria', how='inner')
    paradox_df = paradox_df[paradox_df['Categoria'].notna()]
    
    # 1. Scatter de paradoja con CUADRANTES
    # Calcular medianas para definir cuadrantes
    median_stock = paradox_df['Stock_Total'].median()
    median_nps = paradox_df['NPS_Promedio'].median()
    
    fig_paradox = px.scatter(
        paradox_df,
        x='Stock_Total',
        y='NPS_Promedio',
        size='Feedback_Count',
        color='Rating_Promedio',
        text='Categoria',
        color_continuous_scale='RdYlGn',
        title='Paradoja: Alto Stock vs Sentimiento del Cliente'
    )
    
    # Añadir líneas de cuadrantes
    fig_paradox.add_hline(y=median_nps, line_dash="dash", line_color="gray", 
                         annotation_text=f"NPS Mediano: {median_nps:.1f}")
    fig_paradox.add_vline(x=median_stock, line_dash="dash", line_color="gray",
                         annotation_text=f"Stock Mediano: {median_stock:,.0f}")
    
    # Añadir anotaciones de cuadrantes
    fig_paradox.add_annotation(x=paradox_df['Stock_Total'].max()*0.9, y=paradox_df['NPS_Promedio'].max()*0.9,
                              text="✅ Ideal", showarrow=False, font=dict(size=12, color="green"))
    fig_paradox.add_annotation(x=paradox_df['Stock_Total'].max()*0.9, y=paradox_df['NPS_Promedio'].min()*0.9,
                              text="⚠️ PARADOJA", showarrow=False, font=dict(size=12, color="red"))
    fig_paradox.add_annotation(x=paradox_df['Stock_Total'].min()*1.1, y=paradox_df['NPS_Promedio'].max()*0.9,
                              text="📦 Falta Stock", showarrow=False, font=dict(size=10, color="orange"))
    fig_paradox.add_annotation(x=paradox_df['Stock_Total'].min()*1.1, y=paradox_df['NPS_Promedio'].min()*0.9,
                              text="🔴 Crítico", showarrow=False, font=dict(size=10, color="darkred"))
    
    fig_paradox.update_traces(textposition='top center')
    fig_paradox.update_layout(template='plotly_white', height=500,
                             xaxis_title='Stock Total (unidades)',
                             yaxis_title='NPS Promedio')
    charts['paradoja_stock_nps'] = fig_paradox
    
    # 2. Gráfico de barras comparativo (más claro que heatmap)
    # Normalizar correctamente: Stock 0-1, NPS de -100/+100 a 0-1, Rating de 1-5 a 0-1
    paradox_df['Stock_Norm'] = paradox_df['Stock_Total'] / paradox_df['Stock_Total'].max()
    paradox_df['NPS_Norm'] = (paradox_df['NPS_Promedio'] + 100) / 200  # Convierte -100/+100 a 0-1
    paradox_df['Rating_Norm'] = (paradox_df['Rating_Promedio'] - 1) / 4  # Convierte 1-5 a 0-1
    
    # Crear gráfico de barras agrupadas
    fig_bars = go.Figure()
    
    categorias = paradox_df['Categoria'].tolist()
    
    fig_bars.add_trace(go.Bar(
        name='Stock (normalizado)',
        x=categorias,
        y=paradox_df['Stock_Norm'],
        marker_color='#636EFA',
        text=[f"{v:.0%}" for v in paradox_df['Stock_Norm']],
        textposition='outside'
    ))
    
    fig_bars.add_trace(go.Bar(
        name='NPS (normalizado)',
        x=categorias,
        y=paradox_df['NPS_Norm'],
        marker_color=['#EF553B' if v < 0.5 else '#00CC96' for v in paradox_df['NPS_Norm']],
        text=[f"{v:.0%}" for v in paradox_df['NPS_Norm']],
        textposition='outside'
    ))
    
    fig_bars.add_trace(go.Bar(
        name='Rating (normalizado)',
        x=categorias,
        y=paradox_df['Rating_Norm'],
        marker_color='#AB63FA',
        text=[f"{v:.0%}" for v in paradox_df['Rating_Norm']],
        textposition='outside'
    ))
    
    fig_bars.update_layout(
        title='Comparativa por Categoría: Stock vs Satisfacción',
        barmode='group',
        template='plotly_white',
        height=450,
        yaxis_title='Valor Normalizado (0-100%)',
        xaxis_title='Categoría',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Añadir línea de referencia al 50%
    fig_bars.add_hline(y=0.5, line_dash="dot", line_color="gray", 
                       annotation_text="50% (punto medio)")
    
    charts['matriz_rendimiento'] = fig_bars
    
    return charts


# =============================================================================
# FUNCIÓN DE INTEGRACIÓN CON GROQ
# =============================================================================

def generate_ai_insights(df, kpis, api_key):
    """
    Genera insights usando el modelo Llama-3 de Groq.
    
    Args:
        df: DataFrame filtrado con los datos actuales
        kpis: Diccionario de KPIs calculados
        api_key: API Key de Groq
    
    Returns:
        str: Texto con las recomendaciones estratégicas
    """
    from groq import Groq
    
    # Preparar resumen estadístico
    resumen = f"""
    RESUMEN EJECUTIVO DE TECHLOGISTICS S.A.
    
    MÉTRICAS FINANCIERAS:
    - Ingresos Totales: ${kpis['ingresos_totales']:,.2f} USD
    - Margen Total: ${kpis['margen_total']:,.2f} USD
    - Margen Porcentual Global: {kpis['margen_porcentaje_global']:.1f}%
    - Pérdidas por Margen Negativo: ${kpis['perdidas_margen_negativo']:,.2f} USD
    - Transacciones con Pérdida: {kpis['transacciones_margen_negativo']:,} ({kpis['porcentaje_transacciones_perdida']:.1f}%)
    
    MÉTRICAS LOGÍSTICAS:
    - Tiempo de Entrega Promedio: {kpis['tiempo_entrega_promedio']:.1f} días
    - Entregas Retrasadas: {kpis['entregas_retrasadas']:,} ({kpis['porcentaje_entregas_retrasadas']:.1f}%)
    
    RIESGO DE INVENTARIO:
    - Ventas sin SKU en Catálogo: {kpis['ventas_sku_fantasma']:,}
    - Ingresos en Riesgo: ${kpis['ingresos_sku_fantasma']:,.2f} USD ({kpis['porcentaje_ingresos_fantasma']:.1f}%)
    
    SATISFACCIÓN DEL CLIENTE:
    - NPS Promedio: {kpis['nps_promedio']:.1f}
    - Rating Producto: {kpis['rating_producto_promedio']:.2f}/5
    - Rating Logística: {kpis['rating_logistica_promedio']:.2f}/5
    - Tasa de Tickets de Soporte: {kpis['porcentaje_tickets_soporte']:.1f}%
    
    TRANSACCIONES ANALIZADAS: {len(df):,}
    """
    
    prompt = f"""Eres un consultor senior de análisis de datos para TechLogistics S.A., una empresa de retail tecnológico.

Analiza el siguiente resumen estadístico y genera EXACTAMENTE 3 párrafos de recomendaciones estratégicas:

{resumen}

INSTRUCCIONES:
1. Primer párrafo: Diagnóstico de la situación financiera y márgenes. Identifica los problemas críticos de rentabilidad.
2. Segundo párrafo: Análisis de la operación logística y su impacto en la satisfacción del cliente.
3. Tercer párrafo: Recomendaciones concretas y accionables para los próximos 90 días.

Sé específico con números y porcentajes. Usa un tono ejecutivo y directo.
"""

    try:
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un consultor senior especializado en análisis de datos y estrategia empresarial para empresas de retail tecnológico. Respondes siempre en español."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"Error al generar insights: {str(e)}"


def export_cleaning_report_to_csv(reports, output_path):
    """
    Exporta el reporte de limpieza a un archivo CSV estructurado.
    """
    rows = []
    
    for report in reports:
        # Métricas generales
        row = {
            'Dataset': report['dataset'],
            'Registros_Antes': report['metricas_antes']['registros'],
            'Registros_Despues': report['metricas_despues']['registros'],
            'Health_Score_Antes': report['metricas_antes']['health_score'],
            'Health_Score_Despues': report['metricas_despues']['health_score'],
            'Mejora_Health_Score': report['mejora_health_score'],
            'Completitud_Antes': report['metricas_antes']['completitud'],
            'Completitud_Despues': report['metricas_despues']['completitud'],
            'Unicidad_Antes': report['metricas_antes']['unicidad'],
            'Unicidad_Despues': report['metricas_despues']['unicidad'],
            'Duplicados_Eliminados': report['metricas_antes']['duplicados'],
            'Celdas_Nulas_Antes': report['metricas_antes']['celdas_nulas'],
            'Celdas_Nulas_Despues': report['metricas_despues']['celdas_nulas'],
            'Acciones_Realizadas': '; '.join(report['acciones_realizadas']),
        }
        
        # Agregar información de imputaciones
        for col, info in report.get('imputaciones', {}).items():
            row[f'Imputacion_{col}_Metodo'] = info.get('metodo', '')
            row[f'Imputacion_{col}_Justificacion'] = info.get('justificacion', '')
            row[f'Imputacion_{col}_Valores'] = info.get('valores_imputados', 0)
        
        # Agregar información de outliers
        for col, info in report.get('outliers_detectados', {}).items():
            row[f'Outliers_{col}_Cantidad'] = info.get('cantidad', 0)
        
        rows.append(row)
    
    df_report = pd.DataFrame(rows)
    df_report.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return df_report
