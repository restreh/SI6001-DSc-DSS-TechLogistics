# 🏢 TechLogistics S.A. - Sistema de Soporte a la Decisión (DSS)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Groq](https://img.shields.io/badge/Groq-00D4AA?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)

## 📋 Descripción del Problema

**TechLogistics S.A.** es una empresa ficticia de retail tecnológico que ha detectado:
- 📉 Erosión en márgenes de beneficio
- 👥 Caída drástica en lealtad de clientes
- 🔍 Invisibilidad operativa entre sus sistemas ERP

Este dashboard es un **Sistema de Soporte a la Decisión (DSS)** que transforma el caos de datos en estrategias accionables de recuperación.

## 🎯 Características Principales

### 🔬 Auditoría de Calidad de Datos
- Health Score por dataset (antes/después de limpieza)
- Métricas de completitud, unicidad y validez
- Detección automática de outliers con IQR
- Justificación ética de decisiones de imputación
- Reporte descargable de limpieza

### 🏭 Análisis Operacional
- **Pregunta 1:** Fuga de Capital - SKUs con margen negativo
- **Pregunta 2:** Crisis Logística - Correlación tiempo de entrega vs NPS
- **Pregunta 3:** Venta Invisible - Impacto de SKUs fantasma

### 👥 Análisis de Cliente
- **Pregunta 4:** Paradoja Stock-Satisfacción
- **Pregunta 5:** Bodegas operando a ciegas

### 🤖 Insights con IA
- Recomendaciones estratégicas generadas por Llama-3 (Groq)
- Análisis contextualizado según filtros aplicados

## 📁 Estructura del Proyecto

```
techlogistics_dss/
├── app.py                 # Aplicación principal de Streamlit
├── data_cleaning.py       # Funciones de limpieza y curaduría
├── utils.py               # Utilidades y visualizaciones
├── requirements.txt       # Dependencias del proyecto
├── README.md              # Este archivo
└── datasets/              # Carpeta de datos (crear)
    ├── inventario_central_v2.csv
    ├── transacciones_logistica_v2.csv
    └── feedback_clientes_v2.csv
```

## 🚀 Guía de Instalación

### Requisitos Previos
- Python 3.9 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar o descargar el repositorio**
```bash
git clone <url-del-repositorio>
cd techlogistics_dss
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Colocar los datasets**
   
   Cree una carpeta `datasets/` en el directorio raíz y copie los archivos CSV:
   - `inventario_central_v2.csv`
   - `transacciones_logistica_v2.csv`
   - `feedback_clientes_v2.csv`

5. **Configurar API Key de Groq (opcional, para IA)**
   
   Opción A - Variable de entorno:
   ```bash
   export GROQ_API_KEY="su-api-key"
   ```
   
   Opción B - Archivo `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "su-api-key"
   ```
   
   Opción C - Ingresarla directamente en la interfaz

6. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

7. **Acceder al dashboard**
   
   Abrir en el navegador: `http://localhost:8501`

## 📊 Datasets Utilizados

| Dataset | Registros | Descripción |
|---------|-----------|-------------|
| `inventario_central_v2.csv` | 2,500 | Maestro de productos con stock, costos y lead times |
| `transacciones_logistica_v2.csv` | 10,000 | Histórico de ventas con logística y tiempos de entrega |
| `feedback_clientes_v2.csv` | 4,500 | Voz del cliente con ratings y NPS |

### Problemas Identificados en los Datos
- ⚠️ Categorías inconsistentes (ej: "smart-phone", "Smartphones", "???")
- ⚠️ Ciudades con múltiples formatos (ej: "MED", "Medellín")
- ⚠️ Lead times mixtos (números vs texto)
- ⚠️ Stock negativo
- ⚠️ Edades imposibles (ej: 195 años)
- ⚠️ SKUs fantasma (ventas sin registro en inventario)
- ⚠️ Outliers de costos y tiempos de entrega

## 🔧 Decisiones Técnicas de Limpieza

### Tratamiento de Outliers
- **Método:** Rango Intercuartílico (IQR) con multiplicador 3
- **Justificación:** Robusto a distribuciones no normales, identifica valores extremos sin eliminar variabilidad natural

### Imputaciones

| Variable | Método | Justificación |
|----------|--------|---------------|
| Lead_Time_Dias | Mediana por categoría | Refleja patrones de negocio por tipo de producto |
| Stock_Actual (nulos) | Valor cero | Representa quiebre de stock |
| Costo_Envio | Mediana por ciudad | Costos varían geográficamente |
| Edad_Cliente | Mediana de edades válidas | Representa cliente típico, resistente a valores extremos |

### Normalización Categórica
- Diccionarios de mapeo para ciudades, bodegas y categorías
- Unificación de mayúsculas/minúsculas
- Tratamiento de valores especiales ("???", "N/A")

## 📈 Métricas Clave (KPIs)

### Financieros
- Ingresos Totales
- Margen Total y Porcentual
- Pérdidas por Margen Negativo

### Logísticos
- Tiempo de Entrega Promedio
- % Entregas Retrasadas
- Brecha de Entrega vs Lead Time

### Cliente
- NPS Promedio
- Rating de Producto y Logística
- Tasa de Tickets de Soporte

### Riesgo
- % Ingresos en SKUs Fantasma
- Días promedio sin revisión de stock

## 🛡️ Manejo de Errores

La aplicación incluye:
- Validación de existencia de archivos
- Try/catch en operaciones críticas
- Mensajes informativos de error
- Valores por defecto para datos faltantes

## 🔐 Gestión de Secretos

El API Key de Groq **NUNCA** debe estar en el código. Opciones seguras:
1. Variables de entorno (`GROQ_API_KEY`)
2. Archivo `secrets.toml` de Streamlit
3. Input protegido en la interfaz

## 🌐 Despliegue en la Nube

### Streamlit Community Cloud
1. Subir repositorio a GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar repositorio
4. Configurar secretos en la interfaz de Streamlit Cloud

### Variables de Entorno Requeridas
```
GROQ_API_KEY=tu-api-key-de-groq
```

## 📝 Uso del Dashboard

### Barra Lateral
- **Filtros de Fecha:** Rango temporal de análisis
- **Categorías:** Filtrar por tipo de producto
- **Bodegas:** Filtrar por origen de despacho
- **Ciudades:** Filtrar por destino
- **Canales:** Filtrar por canal de venta
- **Opciones:** Incluir/excluir SKUs fantasma y outliers

### Pestañas Principales
1. **🔬 Auditoría:** Calidad de datos pre/post limpieza
2. **🏭 Operaciones:** Análisis financiero y logístico
3. **👥 Cliente:** Satisfacción y fidelidad
4. **🤖 Insights IA:** Recomendaciones estratégicas

## 🤝 Contribuciones

Actualmente no se aceptan pull requests.


## 📄 Licencia

Este proyecto es parte de un ejercicio académico para el curso de Fundamentos en Ciencia de Datos de la Universidad EAFIT.

## 👨‍💻 Autor

Desarrollado como parte del Challenge 02 del curso de Fundamentos en Ciencia de Datos (SI6001).

**Estudiantes:** Gia Mariana Calle Higuita - José Santiago Molano Perdomo - Juan José Restrepo Higuita
**Docente:** Jorge Iván Padilla-Buriticá  
**Universidad EAFIT** - Periodo 2026-1

---

<div align="center">
  <p>Hecho con ❤️ usando Python, Streamlit y Plotly</p>
</div>
