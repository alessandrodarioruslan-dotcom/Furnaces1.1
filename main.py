import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Configurazione Pagina
st.set_page_config(
    page_title="Furnace Energy Optimizer", page_icon="🔥", layout="wide"
)

DATA_PATH = "data/furnace_data.csv"
MODEL_PATH = "data/furnace_model.pkl"

os.makedirs("data", exist_ok=True)


def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame(
            columns=[
                "Data",
                "Temp_Fondo_C",
                "Temp_Volta_C",
                "Cavato_Tonnellate",
                "Gas_Nm3",
                "Elettrico_kWh",
            ]
        )


def train_model(df):
    if len(df) < 5:
        return None
    X = df[["Temp_Fondo_C", "Temp_Volta_C", "Cavato_Tonnellate"]]
    y = df[["Gas_Nm3", "Elettrico_kWh"]]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


st.title("🔥 Controllo Forno Fusorio & Predizione Energetica")
st.markdown(
    "Sistema d'apprendimento per il monitoraggio dei consumi energetici in base alle condizioni di esercizio del forno."
)

df = load_data()

col_input, col_view = st.columns([1, 1.5])

with col_input:
    st.subheader("📝 Inserimento Dati Giornalieri")
    with st.form("furnace_form"):
        date_val = st.date_input("Data misurazione")
        temp_fondo = st.number_input(
            "Temperatura Fondo (°C)",
            min_value=1000.0,
            max_value=1650.0,
            value=1380.0,
            step=1.0,
        )
        temp_volta = st.number_input(
            "Temperatura Volta (°C)",
            min_value=1000.0,
            max_value=1650.0,
            value=1520.0,
            step=1.0,
        )
        cavato = st.number_input(
            "Cavato Giornaliero (Tonnellate)",
            min_value=0.0,
            max_value=1000.0,
            value=250.0,
            step=5.0,
        )

        st.markdown("---")
        st.caption("Consumi Effettivi Registrati")
        gas = st.number_input(
            "Energia Gas (Nm³)",
            min_value=0.0,
            max_value=100000.0,
            value=18000.0,
            step=100.0,
        )
        elec = st.number_input(
            "Energia Elettrodi (kWh)",
            min_value=0.0,
            max_value=100000.0,
            value=12000.0,
            step=100.0,
        )

        submitted = st.form_submit_button("Salva Misurazione e Riaddestra")

        if submitted:
            new_row = {
                "Data": str(date_val),
                "Temp_Fondo_C": temp_fondo,
                "Temp_Volta_C": temp_volta,
                "Cavato_Tonnellate": cavato,
                "Gas_Nm3": gas,
                "Elettrico_kWh": elec,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_PATH, index=False)
            train_model(df)
            st.success("Dati salvati con successo! Modello aggiornato.")

with col_view:
    st.subheader("🤖 Stima Fabbisogno Energetico (AI)")

    if len(df) >= 5:
        st.info(
            "Imposta i parametri operativi previsti per calcolare la stima del mix energetico ideale:"
        )

        p_temp_fondo = st.slider(
            "Target Temp. Fondo (°C)", 1100.0, 1600.0, 1380.0
        )
        p_temp_volta = st.slider(
            "Target Temp. Volta (°C)", 1100.0, 1600.0, 1520.0
        )
        p_cavato = st.slider(
            "Target Cavato Previsto (Tonnellate)", 0.0, 600.0, 250.0
        )

        model = joblib.load(MODEL_PATH)
        pred = model.predict(
            np.array([[p_temp_fondo, p_temp_volta, p_cavato]])
        )

        st.metric(
            label="Gas Stimato (Nm³)", value=f"{int(pred[0][0]):,} Nm³"
        )
        st.metric(
            label="Elettricità Elettrodi Stimata (kWh)",
            value=f"{int(pred[0][1]):,} kWh",
        )
    else:
        st.warning(
            f"Inserisci almeno {5 - len(df)} altre misurazioni per attivare l'apprendimento automatico."
        )

st.markdown("---")
st.subheader("📊 Storico Registrazioni")
st.dataframe(df, use_container_width=True)
