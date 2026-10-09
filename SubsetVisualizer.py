import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import umap.umap_ as umap

warnings.filterwarnings("ignore", category=UserWarning, module="umap")

st.set_page_config(page_title="KTH Kurslandskap – Dynamisk UMAP", layout="wide")


def skapa_omradeslista(value):
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]

# 1. Läs in data och embeddings

@st.cache_data
def load_base_data_and_embeddings():
    if not os.path.exists("embeddings_cache_text_only.npz"):
        raise FileNotFoundError("embeddings_cache_text_only.npz saknas i mappen.")

    cache = np.load("embeddings_cache_text_only.npz", allow_pickle=True)
    embeddings = cache["embeddings"]
    n_embeddings = len(embeddings)

    json_file_name = "kth_courses_map_text_only.json"

    if os.path.exists(json_file_name):
        df = pd.read_json(json_file_name)
    elif os.path.exists("Kursdata.parquet"):
        df = pd.read_parquet("Kursdata.parquet")
    else:
        raise FileNotFoundError(
            "Hittade varken .json, Kursdata.parquet eller Kursdata.csv"
        )

    # Rensa eventuella mellanslag i kolumnnamn
    df.columns = df.columns.astype(str).str.strip()

    if "Omradeslista" not in df.columns:
        omrade_col = "Huvudområde" if "Huvudområde" in df.columns else "Huvudområden"
        if omrade_col in df.columns:
            df["Huvudområde"] = df[omrade_col]
            df["Omradeslista"] = df["Huvudområde"].apply(
                skapa_omradeslista
            )
        else:
            df["Huvudområde"] = "Okänt"
            df["Omradeslista"] = [[] for _ in range(len(df))]

    if "Primart_Omrade" not in df.columns:
        df["Primart_Omrade"] = df["Omradeslista"].apply(
            lambda lst: lst[0] if lst else "Okänt"
        )

    if len(df) != n_embeddings:
        invalid = {"Okänt", "Denna kurs tillhör inget huvudområde."}
        df_filtered = df[~df["Primart_Omrade"].isin(invalid)].reset_index(drop=True)
        if len(df_filtered) == n_embeddings:
            df = df_filtered
        else:
            df = df.iloc[:n_embeddings].reset_index(drop=True)

    return df, embeddings


df, full_embeddings = load_base_data_and_embeddings()

# 2. Karthantering & filtrering

st.sidebar.title("Karthantering")

# Lista alla unika primära områden
alla_primara_omraden = sorted(
    [o for o in df["Primart_Omrade"].unique() if o and o != "Okänt"]
)

default_kandidater = ["Datalogi och datateknik", "Matematik"]
default_selection = [s for s in default_kandidater if s in alla_primara_omraden]
if not default_selection:
    default_selection = alla_primara_omraden[:2]

selected_areas = st.sidebar.multiselect(
    "Välj ämnesområden för lokal UMAP-projektion:",
    options=alla_primara_omraden,
    default=default_selection,
)

st.sidebar.markdown("### Visningsinställningar")
show_points = st.sidebar.checkbox(
    "Visa individuella kurser (punkter)", value=True
)
show_blobs = st.sidebar.checkbox(
    "Visa områdes-blobbar (DBSCAN + Hull)", value=True
)

blob_granularity = st.sidebar.slider(
    "Blob-uppdelning (lägre = fler sub-hulls)",
    min_value=0.5,
    max_value=4.0,
    value=1.5,
    step=0.25,
    help="Ett lägre värde gör DBSCAN känsligare och skapar separata hulls för sub-områden. Ett högre värde klumpar ihop allt till ett stort område.",
)

st.sidebar.markdown("### UMAP-parametrar")
n_neighbors = st.sidebar.slider(
    "n_neighbors (lokal vs global struktur)", min_value=5, max_value=50, value=20
)
min_dist = st.sidebar.slider(
    "min_dist (klumpbildning)",
    min_value=0.05,
    max_value=0.9,
    value=0.35,
    step=0.05,
)

if not selected_areas:
    st.warning("Välj minst ett ämnesområde i menyn.")
    st.stop()

#
# 3. Filtrera enbart på Primart_Omrade

subset_mask = df["Primart_Omrade"].isin(selected_areas).to_numpy()
subset_df = df[subset_mask].copy().reset_index(drop=True)
subset_embeddings = full_embeddings[subset_mask]

if len(subset_df) < 5:
    st.warning(
        f"Det valda urvalet har bara {len(subset_df)} kurser (minst 5 krävs för UMAP)."
    )
    st.stop()

# 4. Lokal UMAP med cache

@st.cache_data(show_spinner=False)
def berakna_umap(
    features: np.ndarray, n_neighbors: int, min_dist: float
) -> np.ndarray:
    effective_k = min(n_neighbors, len(features) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_k,
        min_dist=min_dist,
        n_epochs=100,
        random_state=42,
        low_memory=False,
    )
    return reducer.fit_transform(features)


with st.spinner(f"Beräknar lokal UMAP för {len(subset_df)} kurser..."):
    # Normalisera till enhetsvektorer
    embeddings_norm = subset_embeddings / np.linalg.norm(
        subset_embeddings, axis=1, keepdims=True
    )

    # Kör UMAP
    coords = berakna_umap(embeddings_norm, n_neighbors, min_dist)
    subset_df["X_koordinat"] = coords[:, 0]
    subset_df["Y_koordinat"] = coords[:, 1]

#
# 5. Sökfunktion för specifik kurs

st.sidebar.markdown("### Sök kurs")
subset_df["Sokstrang"] = (
    subset_df["Kurskod"].astype(str) + " - " + subset_df["Kursnamn"].astype(str)
)

valda_kurs_options = ["(Ingen vald)"] + sorted(subset_df["Sokstrang"].tolist())
vald_kurs_str = st.sidebar.selectbox(
    "Sök eller välj en kurs:", options=valda_kurs_options, index=0
)

vald_kurs_rad = None
if vald_kurs_str != "(Ingen vald)":
    vald_kurs_rad = subset_df[subset_df["Sokstrang"] == vald_kurs_str].iloc[0]
    st.sidebar.info(
        f"**Vald kurs:** {vald_kurs_rad['Kurskod']}\n\n"
        f"**Namn:** {vald_kurs_rad['Kursnamn']}\n\n"
        f"**Område:** {vald_kurs_rad['Primart_Omrade']}\n\n"
        f"**Nivå:** {vald_kurs_rad.get('Utbildningsnivå', 'Okänd')}"
    )

# 6. Skapa Plotly-graf

palett = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
color_map = {
    omrade: palett[i % len(palett)] for i, omrade in enumerate(selected_areas)
}

fig = go.Figure()

# Rita ut individuella kurspunkter om på
if show_points:
    fig_scatter = px.scatter(
        subset_df,
        x="X_koordinat",
        y="Y_koordinat",
        color="Primart_Omrade",
        color_discrete_map=color_map,
        hover_name="Kurskod",
        hover_data=[
            c for c in ["Kursnamn", "Utbildningsnivå"] if c in subset_df.columns
        ],
        opacity=0.8,
    )
    for trace in fig_scatter.data:
        trace.legendgroup = trace.name
        fig.add_trace(trace)

# Rita blobbar per område
if show_blobs and len(subset_df) >= 10:
    pts_all = subset_df[["X_koordinat", "Y_koordinat"]].values
    k_nn = min(5, len(pts_all) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_nn + 1).fit(pts_all)
    distances, _ = nbrs.kneighbors(pts_all)

    # Justera eps enligt användaren
    base_dist = np.median(distances[:, 1:])
    eps = base_dist * blob_granularity

    for omrade in selected_areas:
        omrade_df = subset_df[subset_df["Primart_Omrade"] == omrade]
        pts = omrade_df[["X_koordinat", "Y_koordinat"]].values

        if len(pts) < 4:
            continue

        labels = DBSCAN(eps=eps, min_samples=3).fit_predict(pts)
        c_color = color_map.get(omrade, "#999999")

        # Flagga för att endast visa legenden en gång per område om punkterna är dolda
        omrade_legend_shown = show_points

        for c_id in sorted(set(labels)):
            if c_id == -1:  # Brus / isolerade punkter
                continue
            c_pts = pts[labels == c_id]
            if len(c_pts) < 3:
                continue

            try:
                hull = ConvexHull(c_pts)
                hx = list(c_pts[hull.vertices, 0]) + [c_pts[hull.vertices[0], 0]]
                hy = list(c_pts[hull.vertices, 1]) + [c_pts[hull.vertices[0], 1]]
                
                show_in_legend = not omrade_legend_shown
                if show_in_legend:
                    omrade_legend_shown = True

                fig.add_trace(
                    go.Scatter(
                        x=hx,
                        y=hy,
                        mode="lines",
                        fill="toself",
                        fillcolor=c_color,
                        opacity=0.20,
                        line=dict(
                            color=c_color,
                            width=1.5,
                            dash="dot",
                            shape="spline",
                        ),
                        name=omrade,
                        legendgroup=omrade,
                        showlegend=show_in_legend,
                        hoverinfo="skip",
                    )
                )
            except Exception:
                pass

# Markera den sökta kursen med en stjärna
if vald_kurs_rad is not None:
    fig.add_trace(
        go.Scatter(
            x=[vald_kurs_rad["X_koordinat"]],
            y=[vald_kurs_rad["Y_koordinat"]],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=22,
                color="red",
                line=dict(color="black", width=1.5),
            ),
            text=[f"  <b>{vald_kurs_rad['Kurskod']}</b>"],
            textposition="top right",
            textfont=dict(size=14, color="black"),
            name="Vald kurs",
            showlegend=True,
            hovertext=(
                f"<b>{vald_kurs_rad['Kurskod']}</b><br>"
                f"{vald_kurs_rad['Kursnamn']}<br>"
                f"{vald_kurs_rad['Primart_Omrade']}"
            ),
            hoverinfo="text",
        )
    )

fig.update_layout(
    title=f"Lokal projektion ({len(subset_df)} kurser i {len(selected_areas)} valda områden)",
    xaxis_title="UMAP-1",
    yaxis_title="UMAP-2",
    template="plotly_white",
    height=800,
    margin=dict(l=20, r=20, t=50, b=20),
)

st.plotly_chart(fig, use_container_width=True)