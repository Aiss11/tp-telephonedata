"""
Streamlit — État d'un téléphone portable (TP3, Telephone_data)
Construite sur le modèle d'application fourni par le prof : les objets sauvegardés
dans le notebook sont rechargés, puis le meilleur des sept classifieurs fait la prédiction.

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st

st.set_page_config(page_title="État d'un portable", page_icon="📲", layout="wide")


# ---------- Objets issus du notebook (chargés une seule fois) ----------
@st.cache_resource
def charger_objets():
    encoders = jb.load("encoders.joblib")   # adresse, marque, etat
    uniques = jb.load("uniques.joblib")     # modalités de chaque variable texte
    scaler = jb.load("scaler.joblib")       # StandardScaler
    modele = jb.load("best_model.joblib")   # classifieur retenu (Gradient Boosting)
    return encoders, uniques, scaler, modele


encoders, uniques, scaler, modele = charger_objets()
etats = uniques[2]  # D'occasion / Neuf / Réconditionné / Venant


# ---------- Prédiction pour un téléphone ----------
def Pred_func(prix, adresse, marque, dim_ecr, ram, stockage):
    code_adresse = encoders[0].transform([adresse])[0]
    code_marque = encoders[1].transform([marque])[0]
    # même ordre de colonnes que dans le notebook
    vecteur = np.array([prix, code_adresse, code_marque, dim_ecr, ram, stockage]).reshape(1, -1)
    vecteur_norm = scaler.transform(vecteur)
    classe = modele.predict(vecteur_norm)[0]
    return etats[classe]


# ---------- Prédiction pour un fichier ----------
def Pred_func_csv(fichier):
    tableau = pd.read_csv(fichier)
    resultats = []
    for ligne in tableau.values:
        resultats.append(Pred_func(ligne[0], ligne[1], ligne[2], ligne[3], ligne[4], ligne[5]))
    tableau["etat prédit"] = resultats
    return tableau


# ---------- Barre latérale ----------
with st.sidebar:
    st.header("À propos")
    st.write("Modèle entraîné sur des annonces de téléphones publiées sur CoinAfrique Sénégal. "
             "Sept algorithmes ont été comparés ; le Gradient Boosting, meilleur sur la validation, est utilisé ici.")
    st.caption("Projet Machine Learning — classification (TP3)")

st.title("📲 Neuf, venant, reconditionné ou d'occasion ?")
choix = st.radio("Mode", ["Un téléphone", "Un fichier CSV"], horizontal=True)

if choix == "Un téléphone":
    gauche, droite = st.columns(2)
    with gauche:
        prix = st.number_input("Prix (FCFA)", min_value=0, value=250_000, step=5_000)
        adresse = st.selectbox("Lieu de vente", list(uniques[0]))
        marque = st.selectbox("Marque", list(uniques[1]))
    with droite:
        dim_ecr = st.number_input("Taille de l'écran (pouces)", min_value=0.0, value=6.0, step=0.1)
        ram = st.number_input("Mémoire vive (Go)", min_value=0, value=4, step=1)
        stockage = st.number_input("Stockage (Go)", min_value=0, value=128, step=16)

    if st.button("Lancer la prédiction", type="primary"):
        try:
            st.metric("État estimé", Pred_func(prix, adresse, marque, dim_ecr, ram, stockage))
        except Exception as erreur:
            st.error(f"Prédiction impossible : {erreur}")
else:
    st.info("Colonnes attendues, dans cet ordre : prix, adresse, marque, dim_ecr, ram, stockage.")
    fichier = st.file_uploader("Choisir un fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = Pred_func_csv(fichier)
            st.dataframe(tableau, use_container_width=True)
            st.download_button("Récupérer les résultats (CSV)", tableau.to_csv(index=False).encode("utf-8"),
                               "resultats_telephones.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
