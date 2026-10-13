import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURACIÓN DEL DASHBOARD
# -----------------------------
st.set_page_config(
    page_title="Weather Forecast",
    layout="wide"
)

st.title("📈 Forecast (Historic + SARIMAX + XGB + Hybrid)")


# ---------------------------------------------------
# FUNCIONES
# ---------------------------------------------------

@st.cache_data
def load_station():
    df = pd.read_csv("data/pred_north.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df

@st.cache_data
def load_test_metrics():
    df = pd.read_csv("data/testsets_north.csv")
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

@st.cache_data
def load_enso_scenarios():

    scenario_files = {
        "neutral": "data/pred_north_neutral.csv",
        "nino_moderado": "data/pred_north_nino_moderado.csv",
        "nino_fuerte": "data/pred_north_nino_fuerte.csv",
        "nino_fuerte2": "data/pred_north_nino_fuerte_2.csv",
        "nino_fuerte3": "data/pred_north_nino_fuerte_3.csv",
        "nina_moderada": "data/pred_north_nina_moderada.csv"
    }
    scenario_dfs = {}

    for scen, path in scenario_files.items():

        if os.path.exists(path):

            df_s = pd.read_csv(path)
            df_s["time"] = pd.to_datetime(df_s["time"])

            scenario_dfs[scen] = df_s

    return scenario_dfs

def nse(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - (
        np.sum((y_true - y_pred) ** 2)
        / denominator
    )


def kge(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    std_true = np.std(y_true)
    mean_true = np.mean(y_true)

    if std_true == 0 or mean_true == 0:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / std_true
    beta = np.mean(y_pred) / mean_true

    return 1 - np.sqrt(
        (r - 1) ** 2 +
        (alpha - 1) ** 2 +
        (beta - 1) ** 2
    )


def compute_metrics_test(df):

    metrics = {}

    for model in ["sarimax", "xgb", "hybrid"]:

        y_true = df["real"]
        y_pred = df[model]

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        metrics[model] = {
            "rmse": rmse,
            "mae": mae,
            "nse": nse(y_true, y_pred),
            "kge": kge(y_true, y_pred)
        }

    return metrics

# ---------------------------------------------------
# CARGA PRINCIPAL
# ---------------------------------------------------

df = load_station()

required_cols = [
    "time",
    "historic",
    "sarimax",
    "xgb",
    "hybrid"
]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"Faltan columnas en pred_north.csv: {missing_cols}")
    st.stop()

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2 = st.tabs(
    ["📊 Forecast General", "🌊 Escenarios ENSO"]
)

# ===================================================
# TAB 1
# ===================================================

with tab1:

    st.header("📊 Forecast General")

    # ---------------------------------------------------
    # SIDEBAR
    # ---------------------------------------------------

    st.sidebar.header(
        "Opciones (sólo tab #1 Forecast General)"
    )

    models = st.sidebar.multiselect(
        "Modelos a mostrar",
        ["historic", "sarimax", "xgb", "hybrid"],
        default=["historic", "sarimax", "xgb", "hybrid"]
    )

    min_date = df["time"].min().to_pydatetime()
    max_date = df["time"].max().to_pydatetime()

    date_range = st.sidebar.slider(
        "Rango de fechas",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        key="slider_tab1"
    )

    start, end = date_range

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    df_filtered = df[
        (df["time"] >= start)
        & (df["time"] <= end)
    ]

    # ---------------------------------------------------
    # GRÁFICO PRINCIPAL
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    if "historic" in models:
        ax.plot(
            df_filtered["time"],
            df_filtered["historic"],
            label="Historic",
            color="black"
        )

    if "sarimax" in models:
        ax.plot(
            df_filtered["time"],
            df_filtered["sarimax"],
            label="SARIMAX",
            linestyle="--",
            color="blue"
        )

    if "xgb" in models:
        ax.plot(
            df_filtered["time"],
            df_filtered["xgb"],
            label="XGB",
            linestyle=":",
            color="orange"
        )

    if "hybrid" in models:
        ax.plot(
            df_filtered["time"],
            df_filtered["hybrid"],
            label="Hybrid",
            color="red"
        )

    ax.set_title("Forecast - Estación Norte")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------
    # TABLA
    # ---------------------------------------------------

    st.subheader("Datos filtrados")
    st.dataframe(df_filtered)

    # ---------------------------------------------------
    # TEST SET
    # ---------------------------------------------------

    test_metrics = load_test_metrics()

    st.subheader(
        "📉 Comparación en el Test Set (2016-2026)"
    )

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(
        test_metrics.index,
        test_metrics["real"],
        label="Real",
        color="black"
    )

    ax2.plot(
        test_metrics.index,
        test_metrics["sarimax"],
        label="SARIMAX",
        linestyle="--",
        color="blue"
    )

    ax2.plot(
        test_metrics.index,
        test_metrics["xgb"],
        label="XGB",
        linestyle=":",
        color="orange"
    )

    ax2.plot(
        test_metrics.index,
        test_metrics["hybrid"],
        label="Hybrid",
        color="red"
    )

    ax2.set_title("Comparación de Modelos - Test Set")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Valor real")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)
    plt.close(fig2)

    # ---------------------------------------------------
    # MÉTRICAS
    # ---------------------------------------------------

    metrics = compute_metrics_test(test_metrics)

    st.subheader("📊 Métricas del Test Set")

    col1, col2, col3 = st.columns(3)

    for model, col in zip(
        ["sarimax", "xgb", "hybrid"],
        [col1, col2, col3]
    ):

        m = metrics[model]

        col.metric(
            f"{model.upper()} RMSE",
            f"{m['rmse']:.4f}"
        )

        col.metric(
            f"{model.upper()} MAE",
            f"{m['mae']:.4f}"
        )

    st.subheader("📈 Métricas Hidrológicas")

    col1, col2, col3 = st.columns(3)

    for model, col in zip(
        ["sarimax", "xgb", "hybrid"],
        [col1, col2, col3]
    ):

        m = metrics[model]

        col.metric(
            f"{model.upper()} NSE",
            f"{m['nse']:.4f}"
            if pd.notna(m["nse"])
            else "N/A"
        )

        col.metric(
            f"{model.upper()} KGE",
            f"{m['kge']:.4f}"
            if pd.notna(m["kge"])
            else "N/A"
        )

# ===================================================
# TAB 2
# ===================================================

with tab2:

    st.header("🌊 Escenarios ENSO (Hybrid)")

    scenario_dfs = load_enso_scenarios()

    if len(scenario_dfs) == 0:
        st.warning(
            "No se encontraron archivos de escenarios ENSO."
        )
        st.stop()

    scen_selected = st.multiselect(
        "Selecciona escenarios ENSO",
        options=list(scenario_dfs.keys()),
        default=list(scenario_dfs.keys())
    )

    historic = df[["time", "historic"]].copy()
    historic.set_index("time", inplace=True)

    plot_df = historic.copy()

    for scen in scen_selected:

        df_s = scenario_dfs[scen].copy()

        df_s = df_s.set_index("time")

        plot_df = plot_df.join(
            df_s["hybrid"].rename(scen),
            how="outer"
        )

    st.subheader(
        "Filtrar rango de años para comparar escenarios"
    )

    min_enso_date = plot_df.index.min().to_pydatetime()
    max_enso_date = plot_df.index.max().to_pydatetime()

    enso_range = st.slider(
        "Rango de fechas",
        min_value=min_enso_date,
        max_value=max_enso_date,
        value=(min_enso_date, max_enso_date),
        key="enso_range_slider"
    )

    enso_start, enso_end = enso_range

    plot_df_filtered = plot_df[
        (plot_df.index >= pd.to_datetime(enso_start))
        & (plot_df.index <= pd.to_datetime(enso_end))
    ]

    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.plot(
        plot_df_filtered.index,
        plot_df_filtered["historic"],
        label="Histórico",
        color="black",
        linewidth=2
    )

    colors_scen = {
        "neutral": "green",
        "nino_moderado": "orange",
        "nino_fuerte": "red",
        "nino_fuerte2": "brown",
        "nino_fuerte3": "purple",
        "nina_moderada": "blue"
    }

    for scen in scen_selected:

        ax3.plot(
            plot_df_filtered.index,
            plot_df_filtered[scen],
            label=scen.replace("_", " ").title(),
            linestyle="--",
            color=colors_scen.get(scen, "gray")
        )

    last_historic = historic.index.max()

    if (
        last_historic >= pd.to_datetime(enso_start)
        and
        last_historic <= pd.to_datetime(enso_end)
    ):

        ax3.axvline(
            last_historic,
            color="gray",
            linestyle="--",
            linewidth=1
        )

    ax3.set_title(
        "Escenarios ENSO - Estación Norte"
    )

    ax3.set_xlabel("Fecha")
    ax3.set_ylabel("Precipitación")
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Datos de escenarios ENSO")
    st.dataframe(plot_df_filtered.tail(24))
