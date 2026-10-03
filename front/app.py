import gradio as gr
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "http://localhost:8000"


def get_datasets():
    """Récupère la liste des datasets disponibles"""
    try:
        response = requests.get(f"{API_URL}/datasets")
        return response.json()["datasets"]
    except:
        return ["household", "appliances"]


def get_dataset_info(dataset_name):
    """Récupère les informations sur un dataset"""
    try:
        response = requests.get(f"{API_URL}/datasets/{dataset_name}/info")
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_data_sample(dataset_name, split, n_rows):
    """Récupère un échantillon de données"""
    try:
        response = requests.post(
            f"{API_URL}/data/sample",
            json={"dataset": dataset_name, "split": split, "n_rows": n_rows}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_features(dataset_name):
    """Récupère la liste des features"""
    try:
        response = requests.get(f"{API_URL}/datasets/{dataset_name}/features")
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def load_model(dataset_name, model_type="xgboost"):
    """Charge un modèle dans l'API"""
    try:
        response = requests.post(
            f"{API_URL}/model/load",
            json={"model_type": model_type, "dataset": dataset_name}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def make_prediction(dataset_name, features_dict):
    """Fait une prédiction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"dataset": dataset_name, "features": features_dict}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_comparison_results(dataset_name):
    """Récupère les résultats de comparaison depuis le fichier CSV"""
    try:
        import pandas as pd
        df = pd.read_csv(f"../results/{dataset_name}_comparison.csv")
        return df
    except Exception as e:
        return None


def create_features_dict(hour, dayofweek, month, is_weekend, gap, grp, v, gi):
    """Crée un dictionnaire de features avec les valeurs fournies"""
    import numpy as np
    
    # Encodage cyclique
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)
    
    # Valeurs par défaut pour les features manquantes
    return {
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday": 0,
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
        "dow_sin": float(dow_sin),
        "dow_cos": float(dow_cos),
        "global_active_power_lag_1": gap,
        "global_active_power_lag_2": gap * 1.1,
        "global_active_power_lag_3": gap * 0.9,
        "global_active_power_lag_24": gap * 1.2,
        "global_active_power_lag_48": gap * 1.15,
        "global_active_power_lag_168": gap * 1.1,
        "global_reactive_power_lag_1": grp,
        "voltage_lag_1": v,
        "global_intensity_lag_1": gi,
        "sub_metering_1_lag_1": 10,
        "sub_metering_2_lag_1": 5,
        "sub_metering_3_lag_1": 15,
        "sub_metering_rest_lag_1": 20,
        "global_active_power_rollmean_3": gap,
        "global_active_power_rollstd_3": 0.1,
        "global_active_power_rollmean_6": gap * 1.05,
        "global_active_power_rollstd_6": 0.15,
        "global_active_power_rollmean_24": gap * 1.1,
        "global_active_power_rollstd_24": 0.2
    }


def format_prediction(result):
    """Formate le résultat de prédiction pour affichage"""
    if "error" in result:
        return f"❌ Erreur: {result['error']}"
    
    prediction = result.get("prediction", 0)
    dataset = result.get("dataset", "")
    model = result.get("model_type", "")
    
    # Conversion en Watts pour plus de lisibilité
    watts = prediction * 1000
    
    return f"""
### 🔮 Résultat de la prédiction

| Métrique | Valeur |
|----------|--------|
| **Prédiction** | **{prediction:.4f} kW** |
| **En Watts** | **{watts:.0f} W** |
| **Dataset** | {dataset} |
| **Modèle** | {model} |

💡 Le modèle prédit une consommation de **{prediction:.2f} kWh** pour l'heure suivante.
    """


def format_info(info):
    """Formate les informations du dataset pour affichage"""
    if "error" in info:
        return f"❌ Erreur: {info['error']}"
    
    return f"""
### 📊 Informations Dataset: {info['name'].upper()}

| Métrique | Valeur |
|----------|--------|
| **Variable cible** | `{info['target']}` |
| **Nombre de features** | {info['n_features']} |
| **Lignes train** | {info['train_rows']:,} |
| **Lignes validation** | {info['val_rows']:,} |
| **Lignes test** | {info['test_rows']:,} |
| **Période train** | {info['train_start']} → {info['train_end']} |
| **Période test** | {info['test_start']} → {info['test_end']} |
    """


def format_sample(sample):
    """Formate l'échantillon pour affichage"""
    if "error" in sample:
        return f"❌ Erreur: {sample['error']}", None
    
    df = pd.DataFrame(sample["data"], columns=sample["columns"], index=sample["index"])
    return df.head(sample["shape"][0]), df


def format_features(features):
    """Formate la liste des features"""
    if "error" in features:
        return f"❌ Erreur: {features['error']}"
    
    feature_list = features["features"]
    target = features["target"]
    
    text = f"### 🔧 Features Dataset: {features['dataset'].upper()}\n\n"
    text += f"**Cible:** `{target}`\n\n"
    text += f"**Nombre de features:** {len(feature_list)}\n\n"
    text += "**Liste des features:**\n\n"
    
    for i, feat in enumerate(feature_list, 1):
        text += f"{i}. `{feat}`\n"
    
    return text


def create_time_series_plot(df, target_col):
    """Crée un graphique temporel"""
    if df is None or len(df) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[target_col],
        mode='lines+markers',
        name=target_col,
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f"Évolution de {target_col}",
        xaxis_title="Date",
        yaxis_title=target_col,
        template="plotly_white",
        height=400
    )
    
    return fig


# Interface Gradio
with gr.Blocks(title="Prédiction Énergie - Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ⚡ Prédiction de la Consommation Énergétique
        
        Interface d'exploration des données pour le projet de prédiction de consommation énergétique.
        """
    )
    
    with gr.Tabs():
        # Tab 1: Vue d'ensemble
        with gr.Tab("📋 Vue d'ensemble"):
            gr.Markdown("### Sélectionnez un dataset pour voir ses informations")
            
            with gr.Row():
                dataset_dropdown = gr.Dropdown(
                    choices=get_datasets(),
                    value="household",
                    label="Dataset",
                    info="Choisissez le dataset à explorer"
                )
                info_btn = gr.Button("🔍 Afficher infos", variant="primary")
            
            info_output = gr.Markdown(label="Informations")
            
            info_btn.click(
                fn=lambda x: format_info(get_dataset_info(x)),
                inputs=dataset_dropdown,
                outputs=info_output
            )
        
        # Tab 2: Exploration des données
        with gr.Tab("🔍 Exploration des données"):
            gr.Markdown("### Visualisez les données brutes")
            
            with gr.Row():
                dataset_dropdown2 = gr.Dropdown(
                    choices=get_datasets(),
                    value="household",
                    label="Dataset"
                )
                split_dropdown = gr.Dropdown(
                    choices=["train", "val", "test"],
                    value="train",
                    label="Split"
                )
                n_rows_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=10,
                    step=5,
                    label="Nombre de lignes"
                )
                sample_btn = gr.Button("📊 Charger données", variant="primary")
            
            with gr.Row():
                sample_output = gr.Dataframe(label="Données")
                df_state = gr.State(None)  # Stocker le DataFrame pour le graphique
            
            with gr.Row():
                plot_btn = gr.Button("📈 Afficher graphique", variant="secondary")
                plot_output = gr.Plot(label="Graphique temporel")
            
            sample_btn.click(
                fn=lambda d, s, n: format_sample(get_data_sample(d, s, n)),
                inputs=[dataset_dropdown2, split_dropdown, n_rows_slider],
                outputs=[sample_output, df_state]
            )
            
            plot_btn.click(
                fn=lambda df, d: create_time_series_plot(df, get_dataset_info(d)["target"]) if df is not None and len(df) > 0 else None,
                inputs=[df_state, dataset_dropdown2],
                outputs=plot_output
            )
        
        # Tab 3: Features
        with gr.Tab("🔧 Features"):
            gr.Markdown("### Liste complète des features disponibles")
            
            with gr.Row():
                dataset_dropdown3 = gr.Dropdown(
                    choices=get_datasets(),
                    value="household",
                    label="Dataset"
                )
                features_btn = gr.Button("📋 Afficher features", variant="primary")
            
            features_output = gr.Markdown(label="Features")
            
            features_btn.click(
                fn=lambda x: format_features(get_features(x)),
                inputs=dataset_dropdown3,
                outputs=features_output
            )
        
        # Tab 4: Prédiction
        with gr.Tab("🤖 Prédiction"):
            gr.Markdown("### Faites une prédiction de consommation énergétique")
            
            with gr.Row():
                pred_dataset = gr.Dropdown(
                    choices=["household", "appliances"],
                    value="household",
                    label="Dataset"
                )
                pred_model = gr.Dropdown(
                    choices=["xgboost", "random_forest"],
                    value="xgboost",
                    label="Modèle"
                )
                load_model_btn = gr.Button("📥 Charger modèle", variant="primary")
            
            load_model_output = gr.Markdown(label="État du modèle")
            
            # Formulaire de prédiction simplifié (features principales)
            gr.Markdown("### Entrez les valeurs des features principales")
            
            with gr.Row():
                hour = gr.Slider(0, 23, value=14, step=1, label="Heure (0-23)")
                dayofweek = gr.Slider(0, 6, value=2, step=1, label="Jour de la semaine (0=Lundi)")
                month = gr.Slider(1, 12, value=9, step=1, label="Mois (1-12)")
                is_weekend = gr.Radio([0, 1], value=0, label="Week-end ?")
            
            with gr.Row():
                global_active_power_lag_1 = gr.Number(value=0.5, label="Consommation lag 1h (kW)")
                global_reactive_power_lag_1 = gr.Number(value=0.1, label="Puissance réactive lag 1h (kVAR)")
                voltage_lag_1 = gr.Number(value=235, label="Tension lag 1h (V)")
                global_intensity_lag_1 = gr.Number(value=0.5, label="Intensité lag 1h (A)")
            
            predict_btn = gr.Button("🔮 Prédire", variant="primary", size="lg")
            
            prediction_output = gr.Markdown(label="Résultat de la prédiction")
            
            load_model_btn.click(
                fn=lambda d, m: f"**Modèle chargé:** {load_model(d, m).get('status', 'Erreur')}",
                inputs=[pred_dataset, pred_model],
                outputs=load_model_output
            )
            
            predict_btn.click(
                fn=lambda d, h, dow, m, we, gap, grp, v, gi: format_prediction(
                    make_prediction(d, create_features_dict(h, dow, m, we, gap, grp, v, gi))
                ),
                inputs=[pred_dataset, hour, dayofweek, month, is_weekend, 
                       global_active_power_lag_1, global_reactive_power_lag_1, 
                       voltage_lag_1, global_intensity_lag_1],
                outputs=prediction_output
            )
        
        # Tab 5: Comparaison des modèles
        with gr.Tab("📊 Comparaison des modèles"):
            gr.Markdown("### Résultats de comparaison des modèles")
            
            with gr.Row():
                comp_dataset = gr.Dropdown(
                    choices=["household", "appliances"],
                    value="household",
                    label="Dataset"
                )
                comp_btn = gr.Button("📈 Afficher comparaison", variant="primary")
            
            comp_output = gr.Dataframe(label="Tableau comparatif")
            
            comp_btn.click(
                fn=lambda d: get_comparison_results(d),
                inputs=comp_dataset,
                outputs=comp_output
            )
        
        # Tab 6: API Status
        with gr.Tab("🔌 API Status"):
            gr.Markdown("### État de l'API")
            
            status_btn = gr.Button("🔄 Vérifier API", variant="primary")
            status_output = gr.JSON(label="Réponse API")
            
            status_btn.click(
                fn=lambda: requests.get(f"{API_URL}/").json(),
                outputs=status_output
            )
    
    gr.Markdown(
        """
        ---
        💡 **Conseils:**
        - **Vue d'ensemble** : Informations sur les datasets
        - **Exploration des données** : Visualiser les données brutes
        - **Features** : Liste complète des variables explicatives
        - **Prédiction** : Faire une prédiction en temps réel
        - **Comparaison des modèles** : Voir les performances des différents modèles
        - **API Status** : Vérifier l'état de l'API
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
