import json
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import joblib

# =========================
# 1. Cargar dataset y modelo
# =========================

# Dataset limpio para el dashboard
df = pd.read_csv("student_clean.csv")

# Dataset original para construir instancias compatibles con el modelo
df_modelo = pd.read_csv("student-mat.csv", sep=";")

# Modelo entrenado y guardado
modelo = joblib.load("modelo_g3.pkl")

# Columnas exactas que el modelo espera
MODEL_FEATURES = modelo.named_steps["preprocess"].feature_names_in_.tolist()

# Convertir sexo numérico a texto para mostrarlo mejor en el dashboard
df["sexo_texto"] = df["sex"].map({
    0: "Femenino",
    1: "Masculino"
})

# Crear columna de estado académico para visualización
df["aprobado"] = df["G3"].apply(lambda x: "Aprobado" if x >= 10 else "Reprobado")

# Columnas numéricas para matriz de correlación
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# =========================
# Datos reales para mapa de Panamá
# =========================

gdf_distritos = gpd.read_file("mapa_panama/panama_distritos.geojson")
datos_distritos = pd.read_csv("mapa_panama/datos_distritos.csv")

# Limpiar nombres para evitar problemas por espacios o codificación
gdf_distritos["shapeName_clean"] = gdf_distritos["shapeName"].astype(str).str.strip()
datos_distritos["distrito_clean"] = datos_distritos["distrito"].astype(str).str.strip()

# Unir el GeoJSON con el CSV usando el nombre limpio del distrito
gdf_mapa = gdf_distritos.merge(
    datos_distritos,
    left_on="shapeName_clean",
    right_on="distrito_clean",
    how="left"
)

# Si algún distrito no logra unirse, se rellena con 0 para que el mapa no quede sin colores
gdf_mapa["poblacion_estimada"] = gdf_mapa["poblacion_estimada"].fillna(0)
gdf_mapa["indice_sociodemografico"] = gdf_mapa["indice_sociodemografico"].fillna(0)

# Convertir a GeoJSON para Plotly
geojson_distritos = json.loads(gdf_mapa.to_json())

# =========================
# 2. Crear aplicación Dash
# =========================

app = Dash(__name__)
server = app.server


app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "margin": "30px",
        "backgroundColor": "#f7f7f7",
    },
    children=[
        html.H1(
            "Rendimiento Estudiantil",
            style={"textAlign": "center"}
        ),

        html.P(
            "Este dashboard permite analizar el rendimiento académico de estudiantes "
            "y utilizar un modelo de Machine Learning para predecir la nota final G3.",
            style={"textAlign": "center", "fontSize": "18px"},
        ),

        html.Hr(),

        html.H2("Filtros interactivos"),

        html.Div(
            style={"display": "flex", "gap": "30px", "marginBottom": "30px"},
            children=[
                html.Div(
                    style={"width": "40%"},
                    children=[
                        html.Label("Filtrar por sexo:"),
                        dcc.Dropdown(
                            id="sexo-dropdown",
                            options=[
                                {"label": "Todos", "value": "Todos"},
                                {"label": "Femenino", "value": 0},
                                {"label": "Masculino", "value": 1},
                            ],
                            value="Todos",
                            clearable=False,
                        ),
                    ],
                ),

                html.Div(
                    style={"width": "50%"},
                    children=[
                        html.Label("Edad máxima:"),
                        dcc.Slider(
                            id="edad-slider",
                            min=int(df["age"].min()),
                            max=int(df["age"].max()),
                            step=1,
                            value=int(df["age"].max()),
                            marks={
                                int(x): str(int(x))
                                for x in sorted(df["age"].unique())
                            },
                        ),
                    ],
                ),
            ],
        ),

        html.H2("Visualizaciones del dataset"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "25px",
            },
            children=[
                dcc.Graph(id="grafico-distribucion-g3"),
                dcc.Graph(id="grafico-aprobados"),
                dcc.Graph(id="grafico-edad-g3"),
                dcc.Graph(id="grafico-correlacion"),
            ],
        ),

        html.Hr(),

        html.H2("Predicción de nota final G3"),

        html.P(
            "Ingrese los valores del estudiante. El modelo seleccionado es "
            "Random Forest Regressor, que obtuvo el mejor desempeño con R² = 0.8075.",
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, 1fr)",
                "gap": "15px",
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px",
            },
            children=[
                html.Div([
                    html.Label("Edad"),
                    dcc.Input(
                        id="input-age",
                        type="number",
                        value=16,
                        min=15,
                        max=22,
                    ),
                ]),

                html.Div([
                    html.Label("Educación de la madre Medu (0-4)"),
                    dcc.Input(
                        id="input-Medu",
                        type="number",
                        value=2,
                        min=0,
                        max=4,
                    ),
                ]),

                html.Div([
                    html.Label("Educación del padre Fedu (0-4)"),
                    dcc.Input(
                        id="input-Fedu",
                        type="number",
                        value=2,
                        min=0,
                        max=4,
                    ),
                ]),

                html.Div([
                    html.Label("Tiempo de viaje traveltime (1-4)"),
                    dcc.Input(
                        id="input-traveltime",
                        type="number",
                        value=1,
                        min=1,
                        max=4,
                    ),
                ]),

                html.Div([
                    html.Label("Tiempo de estudio studytime (1-4)"),
                    dcc.Input(
                        id="input-studytime",
                        type="number",
                        value=2,
                        min=1,
                        max=4,
                    ),
                ]),

                html.Div([
                    html.Label("Fracasos previos failures"),
                    dcc.Input(
                        id="input-failures",
                        type="number",
                        value=0,
                        min=0,
                        max=4,
                    ),
                ]),

                html.Div([
                    html.Label("Relación familiar famrel (1-5)"),
                    dcc.Input(
                        id="input-famrel",
                        type="number",
                        value=4,
                        min=1,
                        max=5,
                    ),
                ]),

                html.Div([
                    html.Label("Tiempo libre freetime (1-5)"),
                    dcc.Input(
                        id="input-freetime",
                        type="number",
                        value=3,
                        min=1,
                        max=5,
                    ),
                ]),

                html.Div([
                    html.Label("Salidas goout (1-5)"),
                    dcc.Input(
                        id="input-goout",
                        type="number",
                        value=3,
                        min=1,
                        max=5,
                    ),
                ]),

                html.Div([
                    html.Label("Alcohol diario Dalc (1-5)"),
                    dcc.Input(
                        id="input-Dalc",
                        type="number",
                        value=1,
                        min=1,
                        max=5,
                    ),
                ]),

                html.Div([
                    html.Label("Alcohol fin de semana Walc (1-5)"),
                    dcc.Input(
                        id="input-Walc",
                        type="number",
                        value=1,
                        min=1,
                        max=5,
                    ),
                ]),

                html.Div([
                    html.Label("Salud health (1-5)"),
                    dcc.Input(
                        id="input-health",
                        type="number",
                        value=3,
                        min=1,
                        max=5,
                    ),
                ]),

                html.Div([
                    html.Label("Ausencias"),
                    dcc.Input(
                        id="input-absences",
                        type="number",
                        value=4,
                        min=0,
                        max=100,
                    ),
                ]),

                html.Div([
                    html.Label("Primera nota G1"),
                    dcc.Input(
                        id="input-G1",
                        type="number",
                        value=10,
                        min=0,
                        max=20,
                    ),
                ]),

                html.Div([
                    html.Label("Segunda nota G2"),
                    dcc.Input(
                        id="input-G2",
                        type="number",
                        value=10,
                        min=0,
                        max=20,
                    ),
                ]),
            ],
        ),

        html.Br(),

        html.Button(
            "Predecir G3",
            id="boton-predecir",
            n_clicks=0,
            style={
                "padding": "12px 25px",
                "fontSize": "16px",
                "backgroundColor": "#222",
                "color": "white",
                "border": "none",
                "borderRadius": "8px",
                "cursor": "pointer",
            },
        ),

        html.H2(
            id="resultado-prediccion",
            style={"marginTop": "20px"}
        ),

        html.Hr(),

        html.H2("Mapa interactivo de Panamá"),

html.P(
    "Mapa interactivo de Panamá a nivel distrital. Los colores representan "
    "la población estimada por distrito y se incluye una gráfica resumen "
    "con los distritos de mayor valor."
),

dcc.Graph(id="mapa-panama"),

dcc.Graph(id="grafico-resumen-mapa"),
    ],
)


# =========================
# 3. Callback de gráficos
# =========================

@app.callback(
    Output("grafico-distribucion-g3", "figure"),
    Output("grafico-aprobados", "figure"),
    Output("grafico-edad-g3", "figure"),
    Output("grafico-correlacion", "figure"),
    Input("sexo-dropdown", "value"),
    Input("edad-slider", "value"),
)
def actualizar_graficos(sexo, edad_max):
    data = df.copy()

    if sexo != "Todos":
        data = data[data["sex"] == sexo]

    data = data[data["age"] <= edad_max]

    fig1 = px.histogram(
        data,
        x="G3",
        nbins=20,
        title="Distribución de la nota final G3",
        labels={"G3": "Nota final G3"},
    )

    aprobados = data["aprobado"].value_counts().reset_index()
    aprobados.columns = ["Estado", "Cantidad"]

    fig2 = px.bar(
        aprobados,
        x="Estado",
        y="Cantidad",
        title="Cantidad de aprobados y reprobados",
        labels={
            "Estado": "Estado académico",
            "Cantidad": "Cantidad",
        },
    )

    fig3 = px.scatter(
        data,
        x="age",
        y="G3",
        color="sexo_texto",
        title="Relación entre edad y nota final G3",
        labels={
            "age": "Edad",
            "G3": "Nota final G3",
            "sexo_texto": "Sexo",
        },
    )

    corr = data[numeric_cols].corr()

    fig4 = px.imshow(
        corr,
        text_auto=True,
        title="Mapa de correlación entre variables numéricas",
        aspect="auto",
    )

    return fig1, fig2, fig3, fig4

# =========================
# Callback del mapa interactivo
# =========================

@app.callback(
    Output("mapa-panama", "figure"),
    Output("grafico-resumen-mapa", "figure"),
    Input("sexo-dropdown", "value"),
)
def actualizar_mapa(_):
    fig_mapa = px.choropleth_mapbox(
        gdf_mapa,
        geojson=geojson_distritos,
        locations="shapeID",
        featureidkey="properties.shapeID",
        color="poblacion_estimada",
        hover_name="shapeName",
        hover_data={
            "poblacion_estimada": True,
            "indice_sociodemografico": True,
        },
        mapbox_style="carto-positron",
        center={"lat": 8.5, "lon": -80.0},
        zoom=6.2,
        opacity=0.85,
        title="Mapa interactivo de distritos de Panamá según población estimada",
        color_continuous_scale="Viridis",
        labels={
            "poblacion_estimada": "Población estimada",
            "indice_sociodemografico": "Índice sociodemográfico",
        }, 
    )

    fig_mapa.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=650,
    )

    top_distritos = datos_distritos.sort_values(
        "poblacion_estimada",
        ascending=False
    ).head(10)

    fig_resumen = px.bar(
        top_distritos,
        x="poblacion_estimada",
        y="distrito",
        orientation="h",
        title="Top 10 distritos con mayor población estimada",
        labels={
            "poblacion_estimada": "Población estimada",
            "distrito": "Distrito",
        },
    )

    fig_resumen.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=500,
    )

    return fig_mapa, fig_resumen

# =========================
# 4. Callback de predicción
# =========================

@app.callback(
    Output("resultado-prediccion", "children"),
    Input("boton-predecir", "n_clicks"),
    State("input-age", "value"),
    State("input-Medu", "value"),
    State("input-Fedu", "value"),
    State("input-traveltime", "value"),
    State("input-studytime", "value"),
    State("input-failures", "value"),
    State("input-famrel", "value"),
    State("input-freetime", "value"),
    State("input-goout", "value"),
    State("input-Dalc", "value"),
    State("input-Walc", "value"),
    State("input-health", "value"),
    State("input-absences", "value"),
    State("input-G1", "value"),
    State("input-G2", "value"),
)
def predecir_g3(
    n_clicks,
    age,
    Medu,
    Fedu,
    traveltime,
    studytime,
    failures,
    famrel,
    freetime,
    goout,
    Dalc,
    Walc,
    health,
    absences,
    G1,
    G2,
):
    if n_clicks == 0:
        return "Ingrese los valores y presione el botón para generar una predicción."

    try:
        # Se toma una fila real del dataset original para conservar las variables
        # categóricas en el mismo formato que recibió el modelo durante entrenamiento.
        instancia = df_modelo[MODEL_FEATURES].iloc[[0]].copy()

        # Se reemplazan únicamente las variables numéricas que el usuario ingresa.
        instancia.loc[:, "age"] = int(age)
        instancia.loc[:, "Medu"] = int(Medu)
        instancia.loc[:, "Fedu"] = int(Fedu)
        instancia.loc[:, "traveltime"] = int(traveltime)
        instancia.loc[:, "studytime"] = int(studytime)
        instancia.loc[:, "failures"] = int(failures)
        instancia.loc[:, "famrel"] = int(famrel)
        instancia.loc[:, "freetime"] = int(freetime)
        instancia.loc[:, "goout"] = int(goout)
        instancia.loc[:, "Dalc"] = int(Dalc)
        instancia.loc[:, "Walc"] = int(Walc)
        instancia.loc[:, "health"] = int(health)
        instancia.loc[:, "absences"] = int(absences)
        instancia.loc[:, "G1"] = float(G1)
        instancia.loc[:, "G2"] = float(G2)

        prediccion = modelo.predict(instancia)[0]

        return f"Predicción estimada de G3: {prediccion:.2f} / 20"

    except Exception as error:
        return f"Error al generar la predicción: {error}"


# =========================
# 5. Ejecutar servidor
# =========================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
