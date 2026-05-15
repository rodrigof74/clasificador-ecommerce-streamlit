import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(
    page_title="Clasificación de Usuarios en E-commerce",
    page_icon="🛒",
    layout="wide"
)

# =====================================================
# ESTILO GENERAL STREAMLIT (COMPLETO)
# =====================================================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(34, 211, 238, 0.14), transparent 25%),
            radial-gradient(circle at 85% 25%, rgba(20, 184, 166, 0.16), transparent 28%),
            radial-gradient(circle at 50% 85%, rgba(59, 130, 246, 0.12), transparent 26%),
            linear-gradient(135deg, #020617 0%, #0B1120 45%, #111827 100%);
        color: #E5E7EB;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6, p, li, label {
        color: #E5E7EB !important;
    }

    .section-title {
        color: #E0F2FE;
        font-weight: 900;
        font-size: 29px;
        margin-top: 14px;
        margin-bottom: 14px;
        text-shadow: 0 0 10px rgba(34,211,238,0.18);
    }

    .info-box {
        background: rgba(15, 23, 42, 0.84);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-left: 6px solid #22D3EE;
        font-size: 16px;
        line-height: 1.7;
        color: #E0F2FE;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }

    .glass-box {
        background: rgba(15, 23, 42, 0.78);
        backdrop-filter: blur(12px);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(56,189,248,0.25);
        box-shadow: 0 8px 22px rgba(0,0,0,0.25);
        margin-bottom: 18px;
        color: #E5E7EB;
    }

    .success-box {
        background: linear-gradient(135deg, rgba(20,83,45,0.95), rgba(22,101,52,0.88));
        padding: 18px;
        border-radius: 16px;
        border-left: 6px solid #22C55E;
        font-size: 20px;
        font-weight: 800;
        color: #DCFCE7;
        box-shadow: 0 0 20px rgba(34,197,94,0.25);
    }

    .mini-card {
        background: rgba(15, 23, 42, 0.84);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(56, 189, 248, 0.24);
        box-shadow: 0 8px 20px rgba(0,0,0,0.22);
        text-align: center;
        min-height: 125px;
    }

    .mini-title {
        font-size: 14px;
        color: #93C5FD;
        margin-bottom: 8px;
        font-weight: 700;
    }

    .mini-value {
        font-size: 25px;
        font-weight: 900;
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0F172A 100%);
        border-right: 1px solid rgba(34, 211, 238, 0.22);
    }

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(56,189,248,0.25);
        padding: 16px;
        border-radius: 16px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.22);
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }
    
    .stAlert {
        background-color: rgba(15, 23, 42, 0.9);
        border-left: 6px solid #F59E0B;
    }
    
    .prediction-highlight {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #22D3EE, #3B82F6);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        display: inline-block;
    }
    
    /* Estilo para los forms */
    .stForm {
        background: rgba(15, 23, 42, 0.5);
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LÓGICA DE NEGOCIO (Carga y Preprocesamiento)
# =====================================================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load("modelo_clasificador.pkl")
        return modelo
    except FileNotFoundError:
        st.error(
            "No se encontró el archivo modelo_clasificador.pkl. "
            "Verifique que el archivo esté en la misma carpeta que app.py."
        )
        st.stop()
    except Exception as e:
        st.error(
            "No fue posible cargar el modelo. "
            "Revise que la versión de scikit-learn sea compatible con la usada en Colab."
        )
        st.exception(e)
        st.stop()

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================
def validar_datos_sesion(datos: dict) -> list:
    """Valida la lógica de negocio de los datos de entrada."""
    errores = []
    
    if datos['precio_maximo'] < datos['precio_minimo']:
        errores.append("El Precio Máximo no puede ser menor que el Precio Mínimo.")
    
    if datos['productos_unicos'] > datos['productos_vistos']:
        errores.append("Los Productos Únicos no pueden superar a los Productos Vistos.")
    
    return errores


# =====================================================
# CARACTERIZACIÓN COMERCIAL DE SEGMENTOS
# =====================================================
caracterizacion_segmentos = {
    "Exploradores intensivos": {
        "perfil": (
            "Cliente altamente activo dentro del e-commerce. Presenta una navegación intensa, "
            "con varios clics, revisión de múltiples productos y mayor permanencia exploratoria. "
            "Este perfil suele comparar alternativas antes de tomar una decisión."
        ),
        "conducta": (
            "Tiende a revisar varios modelos, colores o categorías. Puede estar evaluando opciones, "
            "comparando precios, buscando disponibilidad o analizando distintas alternativas antes de comprar."
        ),
        "necesidad": (
            "Necesita recomendaciones claras, filtros eficientes, comparadores de productos y una experiencia "
            "de navegación fluida para no abandonar el sitio."
        ),
        "accion": (
            "Aplicar remarketing dinámico, mostrar productos relacionados, destacar productos vistos recientemente "
            "y ofrecer recomendaciones personalizadas según su historial de navegación."
        )
    },
    "Exploradores medios": {
        "perfil": (
            "Cliente con comportamiento de navegación intermedio. No navega de forma superficial, "
            "pero tampoco presenta una exploración excesivamente alta. Representa un usuario con interés moderado."
        ),
        "conducta": (
            "Revisa una cantidad razonable de productos y categorías. Puede estar buscando un producto específico, "
            "pero todavía no muestra una intención tan fuerte como los exploradores intensivos."
        ),
        "necesidad": (
            "Necesita estímulos simples para continuar navegando, como productos destacados, recomendaciones básicas "
            "o mensajes que faciliten avanzar hacia una decisión."
        ),
        "accion": (
            "Mostrar productos populares, recomendaciones por categoría, llamados a la acción claros y beneficios "
            "como despacho, disponibilidad o promociones moderadas."
        )
    },
    "Exploradores moderados": {
        "perfil": (
            "Cliente de baja a moderada exploración. Presenta una navegación más limitada, con menor cantidad de clics "
            "o menor variedad de productos revisados. Puede tratarse de un usuario con intención poco definida."
        ),
        "conducta": (
            "Suele revisar pocos productos o pocas categorías. Puede abandonar rápidamente si no encuentra algo relevante "
            "o si la experiencia de navegación requiere demasiado esfuerzo."
        ),
        "necesidad": (
            "Necesita una experiencia simple, directa y guiada. Es importante reducir la fricción y mostrar opciones "
            "relevantes desde el inicio."
        ),
        "accion": (
            "Usar banners de categorías destacadas, accesos rápidos, productos más vendidos y recomendaciones simples "
            "para aumentar la exploración."
        )
    },
    "Exploradores premium": {
        "perfil": (
            "Cliente con potencial comercial alto. Se asocia a productos de mayor valor, precios promedio más altos "
            "o patrones de navegación que sugieren interés en artículos de mayor ticket."
        ),
        "conducta": (
            "Puede revisar productos de precio superior, modelos específicos o categorías con mayor valor comercial. "
            "No necesariamente busca descuentos, sino productos con mayor atractivo, calidad o diferenciación."
        ),
        "necesidad": (
            "Necesita una experiencia más personalizada, con información clara de atributos, calidad, disponibilidad "
            "y beneficios asociados al producto."
        ),
        "accion": (
            "Mostrar productos premium, recomendaciones personalizadas, colecciones destacadas, beneficios exclusivos "
            "y campañas de alto valor."
        )
    },
    "Exploradores sensibles al precio": {
        "perfil": (
            "Cliente orientado a precio, promociones u oportunidades de ahorro. Su comportamiento puede estar asociado "
            "a la revisión de ofertas, productos de menor precio o comparación de alternativas económicas."
        ),
        "conducta": (
            "Tiende a revisar productos en oferta, comparar precios o buscar opciones más convenientes. "
            "Puede responder bien a descuentos, promociones y mensajes de ahorro."
        ),
        "necesidad": (
            "Necesita ver claramente el beneficio económico: precio rebajado, descuento, promoción, liquidación "
            "o relación precio-conveniencia."
        ),
        "accion": (
            "Destacar ofertas, descuentos, bundles, productos en promoción, mensajes de ahorro y campañas de remarketing "
            "con incentivo de precio."
        )
    }
}


# =====================================================
# RECOMENDACIÓN DE PRODUCTOS Y RANGO DE PRECIO
# =====================================================
def recomendar_productos_y_precio(categoria_principal, precio_promedio):
    """
    Recomienda dos categorías complementarias según la categoría principal revisada
    y clasifica el rango de precio de interés usando cortes basados en la base por sesión.

    Rangos utilizados según precio promedio observado por sesión:
    - Bajo: 18,00 a 38,17 USD
    - Medio: 38,18 a 49,00 USD
    - Alto: 49,01 a 82,00 USD
    """

    recomendaciones_por_categoria = {
        "pantalones": ["blusas", "faldas"],
        "faldas": ["blusas", "pantalones"],
        "blusas": ["faldas", "pantalones"],
        "ofertas": ["pantalones", "blusas"]
    }

    productos_recomendados = recomendaciones_por_categoria.get(
        categoria_principal,
        ["pantalones", "blusas"]
    )

    if precio_promedio <= 38.17:
        rango_precio = "Bajo"
        rango_valores = "18,00 a 38,17 USD"
        descripcion_precio = (
            "Interés probable en productos económicos, ofertas o alternativas de menor precio. "
            "Este rango corresponde al tramo inferior de precios promedio observados en las sesiones."
        )
        estrategia_precio = (
            "Priorizar descuentos, liquidaciones, promociones y productos de entrada. "
            "También conviene destacar mensajes de ahorro o precio conveniente."
        )
    elif precio_promedio <= 49.00:
        rango_precio = "Medio"
        rango_valores = "38,18 a 49,00 USD"
        descripcion_precio = (
            "Interés probable en productos de precio intermedio. "
            "Este rango representa el comportamiento central de precios promedio observados en la base."
        )
        estrategia_precio = (
            "Recomendar productos populares, combinaciones por categoría y promociones moderadas. "
            "Es un rango adecuado para sugerencias equilibradas entre precio, variedad y atractivo comercial."
        )
    else:
        rango_precio = "Alto"
        rango_valores = "49,01 a 82,00 USD"
        descripcion_precio = (
            "Interés probable en productos de mayor valor o mayor atractivo comercial. "
            "Este rango corresponde al tramo superior de precios promedio observados en las sesiones."
        )
        estrategia_precio = (
            "Mostrar productos premium, colecciones destacadas y recomendaciones personalizadas. "
            "También se pueden priorizar atributos de calidad, diseño, exclusividad o mayor valor percibido."
        )

    return {
        "productos": productos_recomendados,
        "rango_precio": rango_precio,
        "rango_valores": rango_valores,
        "descripcion_precio": descripcion_precio,
        "estrategia_precio": estrategia_precio
    }

# =====================================================
# PORTADA FUTURISTA (COMPLETA)
# =====================================================
components.html(
    """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: Arial, Helvetica, sans-serif;
            }

            .hero-box {
                position: relative;
                background:
                    linear-gradient(135deg, rgba(8, 47, 73, 0.96), rgba(15, 23, 42, 0.98)),
                    radial-gradient(circle at 78% 42%, rgba(34, 211, 238, 0.30), transparent 24%);
                padding: 46px;
                border-radius: 28px;
                border: 1px solid rgba(34, 211, 238, 0.40);
                box-shadow:
                    0 0 45px rgba(34, 211, 238, 0.22),
                    0 24px 60px rgba(0, 0, 0, 0.45);
                min-height: 360px;
                overflow: hidden;
                box-sizing: border-box;
            }

            .hero-box:before {
                content: "";
                position: absolute;
                inset: 0;
                background-image:
                    linear-gradient(rgba(34, 211, 238, 0.08) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(34, 211, 238, 0.08) 1px, transparent 1px);
                background-size: 38px 38px;
                pointer-events: none;
            }

            .hero-box:after {
                content: "";
                position: absolute;
                right: -80px;
                top: -80px;
                width: 360px;
                height: 360px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(34,211,238,0.24), transparent 68%);
                filter: blur(2px);
                pointer-events: none;
            }

            .hero-content {
                position: relative;
                z-index: 3;
                max-width: 760px;
            }

            .glow-pill {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 999px;
                border: 1px solid rgba(34, 211, 238, 0.55);
                color: #A5F3FC;
                background: rgba(8, 145, 178, 0.16);
                font-size: 13px;
                font-weight: 800;
                letter-spacing: 1.2px;
                margin-bottom: 20px;
            }

            .main-title {
                font-size: 48px;
                font-weight: 900;
                color: #F8FAFC;
                margin-bottom: 10px;
                line-height: 1.05;
                text-shadow: 0 0 20px rgba(34, 211, 238, 0.45);
            }

            .subtitle {
                font-size: 21px;
                color: #BAE6FD;
                margin-bottom: 6px;
                font-weight: 600;
            }

            .author {
                font-size: 17px;
                color: #67E8F9;
                margin-bottom: 24px;
                font-weight: 500;
            }

            .hero-text {
                color: #DDEAFE;
                font-size: 16px;
                line-height: 1.8;
                max-width: 730px;
            }

            .orb {
                position: absolute;
                border-radius: 50%;
                background:
                    radial-gradient(circle, rgba(34,211,238,0.95), rgba(8,145,178,0.18) 55%, transparent 72%);
                border: 1px solid rgba(103,232,249,0.40);
                box-shadow: 0 0 38px rgba(34,211,238,0.58);
                opacity: 0.88;
                z-index: 1;
            }

            .orb.one {
                width: 180px;
                height: 180px;
                right: 85px;
                top: 90px;
            }

            .orb.two {
                width: 112px;
                height: 112px;
                right: 285px;
                top: 145px;
            }

            .orb.three {
                width: 88px;
                height: 88px;
                right: 200px;
                top: 275px;
            }

            .orb.four {
                width: 92px;
                height: 92px;
                right: 45px;
                top: 260px;
            }

            .connection {
                position: absolute;
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(34,211,238,0.65), transparent);
                box-shadow: 0 0 12px rgba(34,211,238,0.9);
                z-index: 1;
                transform-origin: left center;
            }

            .connection.one {
                width: 160px;
                right: 200px;
                top: 200px;
                transform: rotate(-12deg);
            }

            .connection.two {
                width: 125px;
                right: 115px;
                top: 260px;
                transform: rotate(62deg);
            }

            .connection.three {
                width: 110px;
                right: 80px;
                top: 235px;
                transform: rotate(-55deg);
            }
        </style>
    </head>

    <body>
        <div class="hero-box">
            <div class="orb one"></div>
            <div class="orb two"></div>
            <div class="orb three"></div>
            <div class="orb four"></div>

            <div class="connection one"></div>
            <div class="connection two"></div>
            <div class="connection three"></div>

            <div class="hero-content">
                <div class="glow-pill">MODELO PREDICTIVO · STREAMLIT · DATA SCIENCE</div>

                <div class="main-title">
                    Clasificación de Usuarios en E-commerce
                </div>

                <div class="subtitle">
                    Solemne 2 - Taller de Aplicaciones
                </div>

                <div class="author">
                    <b>Autor:</b> Rodrigo A. Fuentealba L.
                </div>

                <div class="hero-text">
                     Esta aplicación utiliza un modelo de clasificación para identificar segmentos de usuarios
    que navegan en una tienda de ropa por e-commerce.
    <br><br>
    A partir de variables como clics realizados, productos visualizados, precios observados,
    categoría principal, ubicación geográfica y temporalidad de la sesión, el sistema predice
    a qué grupo pertenece cada usuario según su comportamiento de navegación.
    <br><br>
    Esta información permite apoyar decisiones de segmentación comercial, remarketing,
    personalización de ofertas y análisis de perfiles de comportamiento.
                </div>
            </div>
        </div>
    </body>
    </html>
    """,
    height=430
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Menú")

menu = st.sidebar.radio(
    "Seleccione una sección",
    ["Inicio", "Clasificar sesión", "Resultados del modelo"]
)

st.sidebar.divider()

st.sidebar.info(
    "Aplicación desarrollada en Streamlit para publicar el modelo de clasificación "
    "construido en la Solemne 1.\n\n"
    "Permite ingresar una nueva sesión de navegación y predecir el segmento comercial "
    "al que pertenece el usuario."
)

# =====================================================
# PÁGINA INICIO
# =====================================================
if menu == "Inicio":
    st.markdown('<div class="section-title">📋 Resumen del proyecto</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    El proyecto utiliza datos de navegación de una tienda de ropa en línea.
    La unidad de análisis es la sesión de usuario, ya que resume el comportamiento
    observado durante la navegación: clics realizados, productos explorados,
    variedad de categorías, precios consultados y características temporales.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-title">Modelo final</div>
            <div class="mini-value">Regresión Logística Multinomial</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-title">Accuracy</div>
            <div class="mini-value">0.9902</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="mini-card">
            <div class="mini-title">F1 Macro</div>
            <div class="mini-value">0.9859</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="section-title">🎯 Objetivo de la aplicación</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-box">
    Esta aplicación permite:
    <ul>
        <li>Visualizar los resultados principales del clasificador.</li>
        <li>Ingresar manualmente los datos de una nueva sesión de navegación.</li>
        <li>Clasificar al usuario en un segmento comercial.</li>
        <li>Revisar la probabilidad estimada para cada segmento.</li>
        <li>Apoyar decisiones de segmentación, marketing y personalización comercial.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧩 Segmentos comerciales</div>', unsafe_allow_html=True)

    segmentos = pd.DataFrame({
        "Segmento": [
            "Exploradores intensivos",
            "Exploradores medios",
            "Exploradores moderados",
            "Exploradores premium",
            "Exploradores sensibles al precio"
        ],
        "Interpretación": [
            "Usuarios con alta interacción y exploración amplia dentro del sitio.",
            "Usuarios con navegación intermedia y comportamiento de búsqueda moderado.",
            "Usuarios con exploración controlada y menor intensidad de navegación.",
            "Usuarios asociados a productos de mayor valor o patrones de mayor interés comercial.",
            "Usuarios orientados a precios bajos, promociones u ofertas."
        ]
    })

    st.dataframe(segmentos, use_container_width=True, hide_index=True)

# =====================================================
# PÁGINA CLASIFICAR SESIÓN
# =====================================================
elif menu == "Clasificar sesión":
    st.markdown('<div class="section-title">🔎 Clasificar una nueva sesión</div>', unsafe_allow_html=True)

    modelo = cargar_modelo()

    st.markdown("""
    <div class="info-box">
    Complete los datos asociados al comportamiento de navegación del usuario.
    Luego presione el botón <b>Clasificar sesión</b> para identificar el segmento comercial
    al que pertenece según el modelo entrenado.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # =====================================================
    # TEMPORALIDAD FUERA DEL FORMULARIO
    # Esto permite que "¿Fin de semana?" cambie inmediatamente al cambiar el día.
    # =====================================================
    st.subheader("📅 Temporalidad de la sesión")
    col_dia, col_finde = st.columns([1, 1])

    with col_dia:
        dia_semana_principal = st.selectbox(
            "Día principal",
            ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
            help="Día principal en que se desarrolla la sesión de navegación."
        )

    fin_de_semana = dia_semana_principal in ["Sábado", "Domingo"]

    with col_finde:
        st.markdown(
            f"""
            <div class="info-box">
                <b>¿Fin de semana?</b><br>
                {"Sí" if fin_de_semana else "No"}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    with st.form(key="form_clasificacion"):
        st.subheader("📊 Variables numéricas")
        col1, col2, col3 = st.columns(3)

        with col1:
            clics_sesion = st.number_input(
                "Cantidad de clics de la sesión",
                min_value=1,
                value=8,
                step=1,
                help="Cantidad total de clics realizados por el usuario durante la sesión."
            )
            productos_vistos = st.number_input(
                "Total de productos vistos",
                min_value=1,
                value=8,
                step=1,
                help="Total de productos visualizados durante la sesión, contando repeticiones."
            )
            productos_unicos = st.number_input(
                "Modelos de productos distintos revisados",
                min_value=1,
                max_value=int(productos_vistos),
                value=min(6, int(productos_vistos)),
                step=1,
                help=(
                    "Cantidad de modelos diferentes revisados durante la sesión. "
                    "Ejemplo: si el usuario vio Pantalón A, Pantalón A y Pantalón B, "
                    "entonces productos vistos = 3, pero modelos distintos revisados = 2."
                )
            )

        with col2:
            precio_promedio = st.number_input(
                "Precio promedio visto (USD)",
                min_value=0.0,
                value=45.0,
                step=5.0
            )
            precio_maximo = st.number_input(
                "Precio máximo visto (USD)",
                min_value=0.0,
                value=65.0,
                step=5.0
            )
            precio_minimo = st.number_input(
                "Precio mínimo visto (USD)",
                min_value=0.0,
                value=25.0,
                step=5.0
            )

        with col3:
            categorias_unicas = st.number_input(
                "Variedad de productos explorados",
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                help=(
                    "Cantidad de categorías de productos distintas que el usuario revisó en la sesión: "
                    "pantalones, faldas, blusas u ofertas."
                )
            )
            colores_unicos = st.number_input(
                "Variedad de colores explorados",
                min_value=1,
                max_value=14,
                value=3,
                step=1,
                help=(
                    "Cantidad de colores distintos que el usuario revisó durante la sesión.\n\n"
                    "Ejemplos:\n"
                    "- Solo vio productos negros: negro, negro, negro → 1\n"
                    "- Vio negro y azul: negro, azul, negro → 2\n"
                    "- Vio negro, azul y blanco: negro, azul, blanco → 3\n"
                    "- Vio muchos colores distintos: negro, azul, blanco, rojo, gris → 5"
                )
            )
            paginas_unicas = st.number_input(
                "Páginas distintas visitadas",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                help=(
                    "Cantidad de páginas diferentes de la tienda que el usuario visitó durante la sesión.\n\n"
                    "Ejemplos:\n"
                    "- Solo visitó página 1 → 1\n"
                    "- Visitó página 1 y página 2 → 2\n"
                    "- Visitó página 1, página 2 y página 3 → 3\n"
                    "- Visitó páginas 1 a 5 → 5"
                )
            )

        st.subheader("🏷️ Variables categóricas")
        col4, col5 = st.columns(2)

        with col4:
            categoria_principal = st.selectbox(
                "Categoría principal",
                ["pantalones", "faldas", "blusas", "ofertas"],
                help="Categoría predominante de productos revisados durante la sesión."
            )

        with col5:
            continente_principal = st.selectbox(
                "Continente principal",
                [
                    "Europa del Este", "Europa Occidental", "Europa del Norte", "Europa del Sur",
                    "América", "Asia", "Oceanía", "Sin ubicación"
                ],
                help="Zona geográfica principal asociada a la sesión."
            )

        submitted = st.form_submit_button("🚀 Clasificar sesión", use_container_width=True)

    if submitted:
        datos_sesion = {
            "clics_sesion": clics_sesion,
            "productos_vistos": productos_vistos,
            "productos_unicos": productos_unicos,
            "precio_promedio": precio_promedio,
            "precio_maximo": precio_maximo,
            "precio_minimo": precio_minimo,
            "categorias_unicas": categorias_unicas,
            "colores_unicos": colores_unicos,
            "paginas_unicas": paginas_unicas,
            "categoria_principal": categoria_principal,
            "continente_principal": continente_principal,
            "dia_semana_principal": dia_semana_principal,
            "fin_de_semana": fin_de_semana
        }

        errores_validacion = validar_datos_sesion(datos_sesion)
        if errores_validacion:
            for error in errores_validacion:
                st.error(f"⚠️ {error}")
        else:
            nueva_sesion_df = pd.DataFrame([datos_sesion])

            with st.expander("📋 Ver datos ingresados"):
                st.dataframe(nueva_sesion_df, use_container_width=True, hide_index=True)

            try:
                prediccion = modelo.predict(nueva_sesion_df)[0]

                st.markdown(f"""
                <div class="success-box" style="text-align: center;">
                    🎯 Segmento predicho: <span class="prediction-highlight">{prediccion}</span>
                </div>
                """, unsafe_allow_html=True)

                # Caracterización del cliente tipo según el segmento predicho
                info_cliente = caracterizacion_segmentos.get(str(prediccion))

                if info_cliente:
                    st.markdown("### 🧑‍💼 Caracterización del cliente tipo")
                    st.info(
                        "Esta caracterización describe el perfil comercial típico del segmento predicho. "
                        "No identifica a una persona específica, sino el patrón de comportamiento asociado "
                        "a la sesión ingresada."
                    )

                    col_perfil, col_conducta = st.columns(2)

                    with col_perfil:
                        st.markdown("#### 👤 Perfil del cliente")
                        st.write(info_cliente["perfil"])

                    with col_conducta:
                        st.markdown("#### 🧭 Conducta esperada")
                        st.write(info_cliente["conducta"])

                    col_necesidad, col_accion = st.columns(2)

                    with col_necesidad:
                        st.markdown("#### 🎯 Necesidad comercial detectada")
                        st.write(info_cliente["necesidad"])

                    with col_accion:
                        st.markdown("#### 🚀 Acción recomendada")
                        st.write(info_cliente["accion"])


                # Recomendación de productos/categorías y rango de precio de interés
                recomendacion = recomendar_productos_y_precio(categoria_principal, precio_promedio)

                st.markdown("### 🛍️ Productos recomendados y rango de precio de interés")
                st.info(
                    "Estas recomendaciones se basan en la categoría principal revisada y en el precio promedio observado en la sesión. "
                    "No representan una compra final, sino una sugerencia comercial basada en comportamiento de navegación."
                )

                col_rec1, col_rec2, col_precio = st.columns(3)

                with col_rec1:
                    st.metric(
                        "Producto recomendado 1",
                        recomendacion["productos"][0].capitalize()
                    )

                with col_rec2:
                    st.metric(
                        "Producto recomendado 2",
                        recomendacion["productos"][1].capitalize()
                    )

                with col_precio:
                    st.metric(
                        "Rango de precio de interés",
                        f'{recomendacion["rango_precio"]}: {recomendacion["rango_valores"]}'
                    )

                col_desc_precio, col_estrategia_precio = st.columns(2)

                with col_desc_precio:
                    st.markdown("#### 💵 Interpretación del rango de precio")
                    st.write(recomendacion["descripcion_precio"])

                with col_estrategia_precio:
                    st.markdown("#### 🎯 Estrategia comercial sugerida")
                    st.write(recomendacion["estrategia_precio"])

                # =====================================================
                # RADAR DEL PERFIL DE NAVEGACIÓN
                # =====================================================
                st.markdown("### 🕸️ Radar del perfil de navegación")
                st.info(
                    "Este gráfico resume el comportamiento de la sesión ingresada. "
                    "Los valores se muestran en escala 0 a 100 para facilitar la comparación entre variables."
                )

                def normalizar(valor, minimo, maximo):
                    if maximo == minimo:
                        return 0
                    return max(0, min(100, ((valor - minimo) / (maximo - minimo)) * 100))

                df_radar = pd.DataFrame({
                    "Variable": [
                        "Clics",
                        "Productos vistos",
                        "Modelos distintos",
                        "Categorías exploradas",
                        "Colores explorados",
                        "Páginas visitadas"
                    ],
                    "Valor real": [
                        clics_sesion,
                        productos_vistos,
                        productos_unicos,
                        categorias_unicas,
                        colores_unicos,
                        paginas_unicas
                    ],
                    "Valor normalizado": [
                        normalizar(clics_sesion, 1, 50),
                        normalizar(productos_vistos, 1, 50),
                        normalizar(productos_unicos, 1, 50),
                        normalizar(categorias_unicas, 1, 4),
                        normalizar(colores_unicos, 1, 14),
                        normalizar(paginas_unicas, 1, 5)
                    ]
                })

                fig_radar = px.line_polar(
                    df_radar,
                    r="Valor normalizado",
                    theta="Variable",
                    line_close=True,
                    title="Perfil de navegación de la sesión ingresada",
                    hover_data={
                        "Valor real": True,
                        "Valor normalizado": ":.1f"
                    }
                )

                fig_radar.update_traces(
                    fill="toself",
                    line_color="#22D3EE"
                )

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB")
                )

                st.plotly_chart(fig_radar, use_container_width=True)


                if hasattr(modelo, "predict_proba"):
                    probabilidades = modelo.predict_proba(nueva_sesion_df)[0]
                    clases = modelo.classes_

                    df_prob = pd.DataFrame({
                        "Segmento": clases,
                        "Probabilidad": probabilidades
                    }).sort_values("Probabilidad", ascending=False)

                    st.subheader("📈 Probabilidad por segmento")

                    col_a, col_b = st.columns([1, 2])

                    with col_a:
                        st.dataframe(
                            df_prob.style.format({'Probabilidad': '{:.2%}'}),
                            use_container_width=True,
                            hide_index=True
                        )

                    with col_b:
                        fig = px.bar(
                            df_prob,
                            x="Segmento",
                            y="Probabilidad",
                            text=df_prob["Probabilidad"].apply(lambda x: f"{x:.1%}"),
                            title="Probabilidad estimada por segmento"
                        )
                        fig.update_traces(
                            textposition="outside",
                            marker_color="#22D3EE",
                            marker_line_color="#A5F3FC",
                            marker_line_width=1.5
                        )
                        fig.update_layout(
                            yaxis_tickformat=".0%",
                            xaxis_title="Segmento",
                            yaxis_title="Probabilidad",
                            title_font=dict(size=20, color="#E0F2FE"),
                            xaxis=dict(tickangle=-25),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#E5E7EB")
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
<div class="info-box" style="margin-top: 1rem;">
💡 <b>Interpretación comercial:</b> El segmento predicho representa el perfil comercial más probable del usuario
según su intensidad de navegación, productos explorados, precios observados,
categoría consultada, ubicación geográfica y temporalidad de la sesión.
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}", icon="🤖")

# =====================================================
# PÁGINA RESULTADOS
# =====================================================
elif menu == "Resultados del modelo":
    st.markdown('<div class="section-title">📈 Resultados del clasificador</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    El modelo final utilizado fue una Regresión Logística Multinomial.
    La evaluación muestra un desempeño alto y estable para la clasificación de segmentos
    de usuarios según su comportamiento de navegación en el e-commerce.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "0.9902")

    with col2:
        st.metric("F1 Macro", "0.9859")

    with col3:
        st.metric("Balanced Accuracy", "0.9915")

    metricas = pd.DataFrame({
        "Métrica": ["Accuracy", "F1 Macro", "Balanced Accuracy"],
        "Valor": [0.9902, 0.9859, 0.9915]
    })

    fig = px.bar(
        metricas,
        x="Métrica",
        y="Valor",
        text="Valor",
        title="Métricas principales del clasificador"
    )

    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
        marker_color='#22D3EE'
    )

    fig.update_layout(
        yaxis_range=[0, 1.05],
        xaxis_title="Métrica",
        yaxis_title="Valor",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Interpretación técnica")

    st.markdown("""
    - El modelo presenta un desempeño elevado en accuracy.  
    - El F1 Macro indica un buen equilibrio entre los distintos segmentos.  
    - El Balanced Accuracy permite evaluar el rendimiento considerando posibles diferencias en el tamaño de los grupos.
    """)

    st.subheader("💼 Interpretación comercial")

    st.markdown("""
    El clasificador permite transformar el análisis de comportamiento en una herramienta operativa.
    Una nueva sesión de navegación puede ser asignada automáticamente a un segmento comercial,
    facilitando acciones de marketing, remarketing, personalización de ofertas y análisis de perfiles
    de usuarios en el e-commerce.
    """)

