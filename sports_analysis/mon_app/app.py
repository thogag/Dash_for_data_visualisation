import os
import pandas as pd
from dash import Dash, html, dcc, Input, Output, ALL
import dash
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from ydata_profiling import ProfileReport
from flask import send_from_directory
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import io, base64  
import ast
import json
import plotly.graph_objects as go

# Variable globale pour suivre les colonnes indépendantes sélectionnées
selected_x_columns = []

CLUSTER_COLORS = [
    "#1f77b4",  # bleu
    "#ff7f0e",  # orange
    "#2ca02c",  # vert
    "#d62728",  # rouge
    "#9467bd",  # violet
    "#8c564b",  # marron
]

# --- Vérification et lecture du dataset ---
DATA_PATH = "data/toughestsport.csv"
df = pd.read_csv(DATA_PATH, sep=',')
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# La première colonne contient le nom des sports
sport_col = df.columns[0]
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if len(numeric_cols) < 2:
    raise ValueError("Le dataset doit contenir au moins deux colonnes numériques pour PCA et clustering.")
numeric_cols.remove("Rank")
numeric_cols.remove("Total")


# --- Génération du profil Ydata ---
profile = ProfileReport(df, title="Toughest Sport Analysis", explorative=True)
profile.to_file("report.html")  # Ydata report

# --- Vérification existence du rapport manuel ---
if not os.path.exists("rapport.html"):
    with open("rapport.html", "w") as f:
        f.write("<h1>Rapport manuel</h1><p>Contenu de test ici</p>")

# --- Initialisation Dash ---
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Routes Flask ---
@server.route("/report")
def serve_report():
    abs_path = os.path.abspath("report.html")
    directory = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    return send_from_directory(directory, filename)

@server.route("/rapport")
def serve_rapport():
    abs_path = os.path.abspath("rapport.html")
    directory = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    return send_from_directory(directory, filename)

# --- Page principale ---
home_layout = html.Div([
    html.Div("Projet AD : JORRY - GAGNIEU - GUIGARD", className="top-bar"),

    html.Div([
    html.A("Global visualisation with Ydata", href="/report",
           className="menu-button",
           style={"--rgb-color": "0,150,255", "--border-color": "#005a8c"}),

    html.A("Linear Regression", href="/regression-lineaire",
           className="menu-button",
           style={"--rgb-color": "255,80,80", "--border-color": "#8c1f1f"}),

    html.A("PCA", href="/pca",
           className="menu-button",
           style={"--rgb-color": "180,0,255", "--border-color": "#4b006a"}),

    html.A("Correlation", href="/correlation",
           className="menu-button",
           style={"--rgb-color": "0,200,140", "--border-color": "#00664a"}),

    html.A("Clusters", href="/clusters",
           className="menu-button",
           style={"--rgb-color": "255,150,0", "--border-color": "#8a4a00"}),

    html.A("Rapport", href="/rapport",
           className="menu-button",
           style={"--rgb-color": "255,70,170", "--border-color": "#8c0055"}),
    ], className="button-grid")
], className="home-container")



# --- Layout PCA page dédiée ---
pca_layout_page = html.Div([
    html.H2("Analyse en Composantes Principales (PCA)"),

    dcc.Checklist(
        id="pca-label-toggle",
        options=[{"label": "Afficher les noms des sports sur les points", "value": "show"}],
        value=["show"],  # par défaut : COMME AVANT → labels affichés
        style={"margin-bottom": "20px"}
    ),

    dcc.Graph(id="pca-graph-page"),

    html.Br(),
    html.A(
        "Voir le rapport manuel",
        href="/rapport",
        target="_blank",
        className="menu-link"
    )
])


# --- Layout Clusters ---
clusters_layout = html.Div([
    html.H2("Analyse des clusters"),
    html.P("Choisissez les variables à analyser :"),
    html.Div([
        html.Button(col, id={"type": "col-button", "index": col}, n_clicks=0)
        for col in numeric_cols
    ], style={"display": "flex", "flex-wrap": "wrap", "gap": "5px"}),
    html.Br(),
    html.Label("Nombre de clusters :"),
    dcc.Dropdown(
        id="n-clusters-dropdown",
        options=[{"label": str(i), "value": i} for i in range(2, 7)],
        value=3
    ),
    html.Br(),
    dcc.Graph(id="cluster-graph"),

    # ⭐ Ici on affichera le silhouette plot Yellowbrick sous forme d'image
    html.Img(
        id="silhouette-plot",
        style={"maxWidth": "800px", "width": "100%", "marginTop": "30px"}
    )
])


# --- Layout Régression Linéaire ---
linear_regression_layout = html.Div([
    html.H2("Régression Linéaire"),
    html.P("Choisissez une variable dépendante (Y) :"),
    dcc.Dropdown(
        id="y-variable-dropdown",
        options=[{"label": col, "value": col} for col in numeric_cols],
        value=numeric_cols[0]
    ),
    html.Br(),
    html.P("Choisissez les variables indépendantes (X) :"),
    html.Div([
        html.Button(col, id={"type": "x-col-button", "index": col}, n_clicks=0)
        for col in numeric_cols
    ], style={"display": "flex", "flex-wrap": "wrap", "gap": "5px"}),
    html.Br(),
    dcc.Graph(id="linear-regression-graph")
])

# --- Layout Correlation ---
correlation_layout = html.Div([
    html.H2("Analyse des Corrélations"),
    html.P("Heatmap des corrélations entre toutes les colonnes numériques."),
    dcc.Graph(id="correlation-graph")
])

# --- Layout racine multipage ---
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# --- Callback multi-page ---
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/clusters":
        return clusters_layout
    elif pathname == "/regression-lineaire":
        return linear_regression_layout
    elif pathname == "/correlation":
        return correlation_layout
    elif pathname == "/pca":
        return pca_layout_page
    elif pathname == "/" or pathname is None:
        return home_layout
    else:
        return html.H1("404 - Page not found")

# --- Callback PCA ---
# @app.callback(
#     Output("pca-graph", "figure"),
#     Input("pca-graph", "id")
# )
# def update_pca(_):
#     pca = PCA(n_components=2)
#     components = pca.fit_transform(df[numeric_cols])
#     pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
#     pca_df[sport_col] = df[sport_col]  # ajout du nom des sports
#     fig = px.scatter(pca_df, x="PC1", y="PC2", text=sport_col,
#                      title="PCA des sports", hover_name=sport_col)
#     return fig

@app.callback(
    Output("pca-graph-page", "figure"),
    Input("pca-label-toggle", "value"),
)
def update_pca_page(label_options):
    show_labels = label_options is not None and "show" in label_options

    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(df[numeric_cols])

    # Variance expliquée (PC1, PC2)
    explained = pca.explained_variance_ratio_ * 100
    pc1_var = explained[0]
    pc2_var = explained[1]

    # Contributions des variables aux axes
    loadings = pd.DataFrame(pca.components_.T, index=numeric_cols, columns=["PC1", "PC2"])

    top_PC1 = ", ".join(loadings["PC1"].abs().sort_values(ascending=False).head(3).index)
    top_PC2 = ", ".join(loadings["PC2"].abs().sort_values(ascending=False).head(3).index)

    # DataFrame PCA
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df[sport_col] = df[sport_col]

    # Titres des axes : variance + variables dominantes
    axis_titles = {
        "x": f"PC1 ({pc1_var:.1f}% – {top_PC1})",
        "y": f"PC2 ({pc2_var:.1f}% – {top_PC2})",
    }

    # FIGURE
    if show_labels:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            text=sport_col,
            title="PCA des sports",
            hover_name=sport_col,
        )
    else:
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            title="PCA des sports",
            hover_name=sport_col,
        )

    # ➜ MET À JOUR LES TITRES D’AXES (obligatoire)
    fig.update_layout(
        xaxis_title=axis_titles["x"],
        yaxis_title=axis_titles["y"]
    )

    return fig

def make_silhouette_image(X, n_clusters):
    """
    X : np.array de shape (n_samples, n_features) après normalisation
    n_clusters : nombre de clusters K
    Retourne une string 'data:image/png;base64,...' utilisable dans html.Img(src=...)
    """
    # On recrée un modèle KMeans identique à celui du scatter
    model = KMeans(n_clusters=n_clusters, random_state=0)

    # SilhouetteVisualizer va gérer le fit et le calcul des scores
    visualizer = SilhouetteVisualizer(
    model,
    colors=CLUSTER_COLORS[:n_clusters]  # même ordre que Plotly
    )
    visualizer.fit(X)

    # On sauvegarde la figure matplotlib dans un buffer mémoire
    buf = io.BytesIO()
    visualizer.ax.figure.tight_layout()
    visualizer.ax.figure.savefig(buf, format="png", bbox_inches="tight")
    plt.close(visualizer.ax.figure)  # on ferme la figure pour éviter les fuites mémoire
    buf.seek(0)

    # On encode en base64 pour l'envoyer à Dash
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded


# --- Callback clustering ---
selected_columns = []

@app.callback(
    Output("cluster-graph", "figure"),
    Output("silhouette-plot", "src"),  # <- on ajoute la sortie pour l'image
    Input({"type": "col-button", "index": ALL}, "n_clicks"),
    Input("n-clusters-dropdown", "value")
)

def update_clusters(n_clicks_list, n_clusters):
    ctx = dash.callback_context
    global selected_columns

    # aucun événement encore
    if not ctx.triggered:
        fig = px.scatter(title="Cliquez sur deux variables à analyser")
        return fig, None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 🔹 Si c'est un bouton dynamique (ID sous forme de dict JSON)
    if trigger_id.startswith("{"):
        trigger_dict = ast.literal_eval(trigger_id)
        button_index = trigger_dict.get("index")

        if button_index not in selected_columns:
            selected_columns.append(button_index)

        # on garde au max les 2 dernières colonnes
        if len(selected_columns) > 2:
            selected_columns = selected_columns[-2:]

    # Si c'est le dropdown qui a déclenché, on NE change PAS selected_columns

    if len(selected_columns) < 2:
        fig = px.scatter(title="Sélectionnez deux variables pour le clustering")
        return fig, None

    # Normalisation + KMeans comme avant
    df_normalized = df.copy()
    scaler = StandardScaler()
    df_normalized[selected_columns] = scaler.fit_transform(
        df_normalized[selected_columns]
    )
    clusters_normalized = KMeans(
        n_clusters=n_clusters,
        random_state=0
    ).fit_predict(df_normalized[selected_columns])

    # labels numériques (pour calcul / silhouette)
    df_normalized["cluster"] = clusters_normalized
    # labels texte (pour Plotly → couleurs discrètes)
    df_normalized["cluster_label"] = df_normalized["cluster"].astype(str)

    # Figure plotly de clustering avec palette DISCRÈTE
    fig = px.scatter(
        df_normalized,
        x=selected_columns[0],
        y=selected_columns[1],
        color="cluster_label",  # <- on colore avec la version str
        hover_name=sport_col,
        title=f"Clustering ({n_clusters} clusters) sur "
              f"{selected_columns[0]} vs {selected_columns[1]}",
        color_discrete_sequence=CLUSTER_COLORS[:n_clusters],
        category_orders={"cluster_label": [str(i) for i in range(n_clusters)]}
    )

    # (optionnel, pour que la légende soit jolie)
    fig.update_layout(legend_title_text="Cluster")

    # 👉 Silhouette plot Yellowbrick -> image base64
    X_for_sil = df_normalized[selected_columns].values
    silhouette_src = make_silhouette_image(X_for_sil, n_clusters)

    return fig, silhouette_src




# --- Callback Régression Linéaire ---
@app.callback(
    Output("linear-regression-graph", "figure"),
    Input({"type": "x-col-button", "index": ALL}, "n_clicks"),
    Input("y-variable-dropdown", "value")
)
def update_linear_regression(n_clicks_list, y_col):
    global selected_x_columns
    ctx = dash.callback_context

    if not ctx.triggered or y_col is None:
        return go.Figure().update_layout(
            title="Sélectionnez une variable dépendante et au moins une variable indépendante"
        )

    # Détecte quel élément a déclenché
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Si c'est un bouton dynamique
    if trigger_id.startswith("{"):
        button_index = json.loads(trigger_id)['index']

        # Toggle sélection
        if button_index in selected_x_columns:
            selected_x_columns.remove(button_index)
        else:
            selected_x_columns.append(button_index)

    # Si aucune variable indépendante sélectionnée
    if len(selected_x_columns) == 0:
        return go.Figure().update_layout(
            title="Sélectionnez au moins une variable indépendante"
        )

    # Création figure
    fig = go.Figure()
    for x_col in selected_x_columns:
        X = df[[x_col]]
        y = df[y_col]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        fig.add_trace(go.Scatter(
            x=df[x_col], y=y, mode='markers', name=f"{x_col} vs {y_col}"
        ))
        fig.add_trace(go.Scatter(
            x=df[x_col], y=y_pred, mode='lines', name=f"Régression {x_col} -> {y_col}"
        ))

    fig.update_layout(
        title=f"Régression linéaire: {y_col} vs {', '.join(selected_x_columns)}",
        xaxis_title="Variable indépendante",
        yaxis_title=y_col
    )

    return fig



# --- Callback Correlation ---
@app.callback(
    Output("correlation-graph", "figure"),
    Input("url", "pathname")  
)
def update_correlation(_):
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        labels=dict(x="Variable", y="Variable", color="Corrélation"),
        title="Matrice de corrélation"
    )

    fig.update_layout(
        width=900,
        height=800,
        font=dict(size=10),
        title=dict(font=dict(size=20)),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))

    return fig


# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
