#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime


# Configuration de la page
st.set_page_config(
    page_title="Dashboard d'analyse des barrages par région",
    page_icon="💧",
    layout="wide"
)

# Titre du dashboard
st.title("📊 Dashboard d'analyse des barrages par région")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("Barrages_tn.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar pour les filtres
st.sidebar.header("Filtres")

# Sélection des régions
regions = sorted(df['region'].unique())
all_regions = st.sidebar.checkbox("Sélectionner toutes les régions", True)

if all_regions:
    selected_regions = regions
else:
    selected_regions = st.sidebar.multiselect(
        "Sélectionner les régions",
        options=regions,
        default=regions[:3]  # Par défaut, sélectionner les 3 premières régions
    )

if selected_regions:
    filtered_df = df[df['region'].isin(selected_regions)]
else:
    st.sidebar.warning("Veuillez sélectionner au moins une région")
    st.stop()

# Filtrage optionnel par date pour les analyses temporelles
with st.sidebar.expander("Filtrage par date (optionnel)"):
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_filter = st.checkbox("Activer le filtrage par date", False)
    
    if date_filter:
        start_date, end_date = st.date_input(
            "Sélectionner une plage de dates",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        # Convertir en datetime pour la comparaison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]
    else:
        # Utiliser toutes les dates disponibles
        start_date = pd.to_datetime(min_date)
        end_date = pd.to_datetime(max_date)

# Dashboard principal divisé en onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Vue d'ensemble", "Analyse par région", "Comparaison", "Données brutes","Map"])

with tab5:
    st.subheader("📍 Carte des barrages par taux de remplissage")

    # Construction de la figure
    fig = px.scatter_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        color='taux_remplissage',
        size='taux_remplissage',
        hover_name='nom_barrage',
        hover_data='taux_remplissage',
        color_continuous_scale='Blues',
        size_max=15,
        zoom=6,
        mapbox_style='open-street-map'
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

with tab1:
    st.header("Vue d'ensemble des barrages par région")
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    # Dernières données disponibles pour les métriques
    latest_data = filtered_df[filtered_df['Date'] == filtered_df['Date'].max()]
    
    with col1:
        total_capacity = latest_data['capacite_totale_actuelle'].sum()
        st.metric("Capacité totale (Mm³)", f"{total_capacity:.2f}")
    
    with col2:
        current_stock = latest_data['stock_actuel'].sum()
        st.metric("Stock actuel (Mm³)", f"{current_stock:.2f}")
    
    with col3:
        avg_fill_rate = (current_stock / total_capacity) * 100 if total_capacity > 0 else 0
        st.metric("Taux de remplissage moyen", f"{avg_fill_rate:.2f}%")
    
    with col4:
        total_barrages = latest_data['nom_barrage'].nunique()
        st.metric("Nombre de barrages", f"{total_barrages}")
    
    # Carte de chaleur des taux de remplissage par région
    st.subheader("Taux de remplissage moyen par région")
    
    # Agrégation par région
    region_summary = latest_data.groupby('region').agg({
        'stock_actuel': 'sum',
        'capacite_totale_actuelle': 'sum',
        'nom_barrage': 'count'
    }).reset_index()
    
    region_summary['taux_remplissage'] = (region_summary['stock_actuel'] / region_summary['capacite_totale_actuelle']) * 100
    region_summary.rename(columns={'nom_barrage': 'nombre_barrages'}, inplace=True)
    
    # Graphique du taux de remplissage par région
    fig_region = px.bar(
        region_summary.sort_values('taux_remplissage', ascending=False),
        x='region',
        y='taux_remplissage',
        text_auto='.2f',
        color='taux_remplissage',
        color_continuous_scale=px.colors.sequential.Blues,
        labels={'taux_remplissage': 'Taux de remplissage (%)', 'region': 'Région'},
        title="Taux de remplissage moyen par région (%)"
    )
    
    fig_region.update_layout(height=500)
    fig_region.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Graphique 1: Évolution du stock et du taux de remplissage global
    st.subheader("Évolution du stock et du taux de remplissage")
    
    # Calcul des données agrégées par date
    daily_data = filtered_df.groupby('Date').agg({
        'stock_actuel': 'sum',
        'capacite_totale_actuelle': 'sum',
        'apports_journaliers': 'sum',
        'lachers_journaliers': 'sum'
    }).reset_index()
    
    daily_data['taux_remplissage'] = (daily_data['stock_actuel'] / daily_data['capacite_totale_actuelle']) * 100
    
    # Créer un graphique combiné avec Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['stock_actuel'],
            name="Stock (Mm³)",
            line=dict(color='royalblue', width=2)
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Date'],
            y=daily_data['taux_remplissage'],
            name="Taux de remplissage (%)",
            line=dict(color='firebrick', width=2, dash='dot')
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Évolution du stock et du taux de remplissage",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="Stock (Mm³)", secondary_y=False)
    fig.update_yaxes(title_text="Taux de remplissage (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique 2: Répartition des barrages par région
    st.subheader("Répartition des barrages et capacité par région")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Nombre de barrages par région
        barrage_count = latest_data.groupby('region')['nom_barrage'].nunique().reset_index()
        barrage_count.columns = ['region', 'nombre_barrages']
        
        fig_count = px.pie(
            barrage_count,
            values='nombre_barrages',
            names='region',
            title="Nombre de barrages par région",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig_count.update_layout(height=400)
        st.plotly_chart(fig_count, use_container_width=True)
    
    with col2:
        # Capacité totale par région
        capacity_by_region = latest_data.groupby('region')['capacite_totale_actuelle'].sum().reset_index()
        
        fig_capacity = px.pie(
            capacity_by_region,
            values='capacite_totale_actuelle',
            names='region',
            title="Capacité totale par région (Mm³)",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel2
        )
        
        fig_capacity.update_layout(height=400)
        st.plotly_chart(fig_capacity, use_container_width=True)
    
    # Graphique 3: Apports et lâchers journaliers
    st.subheader("Apports et lâchers journaliers")
    
    fig2 = go.Figure()
    
    fig2.add_trace(
        go.Bar(
            x=daily_data['Date'],
            y=daily_data['apports_journaliers'],
            name="Apports journaliers",
            marker_color='lightgreen'
        )
    )
    
    fig2.add_trace(
        go.Bar(
            x=daily_data['Date'],
            y=daily_data['lachers_journaliers'],
            name="Lâchers journaliers",
            marker_color='salmon'
        )
    )
    
    fig2.update_layout(
        barmode='group',
        title_text="Apports et lâchers journaliers",
        xaxis_title="Date",
        yaxis_title="Volume (Mm³)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Analyse détaillée par région")
    
    # Sélection d'une région spécifique pour analyse détaillée
    selected_region = st.selectbox(
        "Sélectionner une région pour l'analyse détaillée",
        options=selected_regions
    )
    
    # Filtrer les données pour la région sélectionnée
    region_df = filtered_df[filtered_df['region'] == selected_region]
    
    # Informations générales sur la région
    latest_region = region_df[region_df['Date'] == region_df['Date'].max()]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de barrages", f"{latest_region['nom_barrage'].nunique()}")
        
    with col2:
        region_capacity = latest_region['capacite_totale_actuelle'].sum()
        st.metric("Capacité totale (Mm³)", f"{region_capacity:.2f}")
        
    with col3:
        region_stock = latest_region['stock_actuel'].sum()
        st.metric("Stock actuel (Mm³)", f"{region_stock:.2f}")
        
    with col4:
        region_rate = (region_stock / region_capacity) * 100 if region_capacity > 0 else 0
        st.metric("Taux de remplissage moyen", f"{region_rate:.2f}%")
    
    # Liste des barrages dans la région
    st.subheader(f"Barrages dans la région de {selected_region}")
    
    barrages_in_region = latest_region[['nom_barrage', 'stock_actuel', 'capacite_totale_actuelle']].copy()
    barrages_in_region['taux_remplissage'] = (barrages_in_region['stock_actuel'] / barrages_in_region['capacite_totale_actuelle']) * 100
    barrages_in_region = barrages_in_region.sort_values('taux_remplissage', ascending=False)
    
    fig_barrages = px.bar(
        barrages_in_region,
        x='nom_barrage',
        y='taux_remplissage',
        text_auto='.2f',
        color='taux_remplissage',
        color_continuous_scale=px.colors.sequential.Blues,
        labels={'taux_remplissage': 'Taux de remplissage (%)', 'nom_barrage': 'Nom du barrage'},
        title=f"Taux de remplissage des barrages dans la région de {selected_region} (%)"
    )
    
    fig_barrages.update_layout(height=500)
    fig_barrages.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_barrages, use_container_width=True)
    
    # Évolution du taux de remplissage pour la région sélectionnée
    st.subheader(f"Évolution du taux de remplissage - Région de {selected_region}")
    
    # Calculer le taux de remplissage quotidien pour la région
    region_daily = region_df.groupby('Date').agg({
        'stock_actuel': 'sum',
        'capacite_totale_actuelle': 'sum',
        'apports_journaliers': 'sum',
        'lachers_journaliers': 'sum'
    }).reset_index()
    
    region_daily['taux_remplissage'] = (region_daily['stock_actuel'] / region_daily['capacite_totale_actuelle']) * 100
    
    fig_region_evolution = px.line(
        region_daily,
        x='Date',
        y='taux_remplissage',
        labels={'taux_remplissage': 'Taux de remplissage (%)', 'Date': 'Date'},
        title=f"Évolution du taux de remplissage - Région de {selected_region}"
    )
    
    fig_region_evolution.update_layout(height=400)
    
    st.plotly_chart(fig_region_evolution, use_container_width=True)
    
    # Bilan hydrique: Apports vs Lâchers
    st.subheader(f"Bilan hydrique: Apports vs Lâchers - Région de {selected_region}")
    
    # Calcul du bilan hydrique cumulé
    region_daily['bilan_journalier'] = region_daily['apports_journaliers'] - region_daily['lachers_journaliers']
    region_daily['bilan_cumule'] = region_daily['bilan_journalier'].cumsum()
    
    fig_bilan = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Ajouter les barres pour apports et lâchers
    fig_bilan.add_trace(
        go.Bar(
            x=region_daily['Date'],
            y=region_daily['apports_journaliers'],
            name="Apports journaliers",
            marker_color='lightgreen'
        ),
        secondary_y=False
    )
    
    fig_bilan.add_trace(
        go.Bar(
            x=region_daily['Date'],
            y=region_daily['lachers_journaliers'],
            name="Lâchers journaliers",
            marker_color='salmon'
        ),
        secondary_y=False
    )
    
    # Ajouter la ligne pour le bilan cumulé
    fig_bilan.add_trace(
        go.Scatter(
            x=region_daily['Date'],
            y=region_daily['bilan_cumule'],
            name="Bilan hydrique cumulé",
            line=dict(color='royalblue', width=2)
        ),
        secondary_y=True
    )
    
    fig_bilan.update_layout(
        barmode='group',
        title_text=f"Bilan hydrique: Apports vs Lâchers - Région de {selected_region}",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode="x unified"
    )
    
    fig_bilan.update_yaxes(title_text="Volume journalier (Mm³)", secondary_y=False)
    fig_bilan.update_yaxes(title_text="Bilan cumulé (Mm³)", secondary_y=True)
    
    st.plotly_chart(fig_bilan, use_container_width=True)
    
    # Analyse saisonnière
    st.subheader(f"Analyse saisonnière - Région de {selected_region}")
    
    # Extraire le mois et créer des agrégations mensuelles
    region_df['mois'] = region_df['Date'].dt.month
    monthly_data = region_df.groupby('mois').agg({
        'apports_journaliers': 'sum',
        'lachers_journaliers': 'sum',
        'stock_actuel': 'mean',
        'capacite_totale_actuelle': 'mean'
    }).reset_index()
    
    monthly_data['taux_remplissage'] = (monthly_data['stock_actuel'] / monthly_data['capacite_totale_actuelle']) * 100
    
    # Convertir les numéros de mois en noms de mois
    month_names = {
        1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
        7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
    }
    monthly_data['nom_mois'] = monthly_data['mois'].map(month_names)
    monthly_data = monthly_data.sort_values('mois')
    
    # Créer un graphique pour l'analyse saisonnière
    fig_saison = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_saison.add_trace(
        go.Bar(
            x=monthly_data['nom_mois'],
            y=monthly_data['apports_journaliers'],
            name="Apports totaux",
            marker_color='lightgreen'
        ),
        secondary_y=False
    )
    
    fig_saison.add_trace(
        go.Bar(
            x=monthly_data['nom_mois'],
            y=monthly_data['lachers_journaliers'],
            name="Lâchers totaux",
            marker_color='salmon'
        ),
        secondary_y=False
    )
    
    fig_saison.add_trace(
        go.Scatter(
            x=monthly_data['nom_mois'],
            y=monthly_data['taux_remplissage'],
            name="Taux de remplissage moyen",
            line=dict(color='royalblue', width=2, dash='dot')
        ),
        secondary_y=True
    )
    
    fig_saison.update_layout(
        barmode='group',
        title_text=f"Analyse saisonnière - Région de {selected_region}",
        xaxis_title="Mois",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode="x"
    )
    
    fig_saison.update_yaxes(title_text="Volume (Mm³)", secondary_y=False)
    fig_saison.update_yaxes(title_text="Taux de remplissage moyen (%)", secondary_y=True)
    
    st.plotly_chart(fig_saison, use_container_width=True)

with tab3:
    st.header("Comparaison entre régions")
    
    # Sélection des régions à comparer
    regions_a_comparer = st.multiselect(
        "Sélectionner les régions à comparer",
        options=selected_regions,
        default=selected_regions[:min(3, len(selected_regions))]  # Par défaut, comparer jusqu'à 3 régions
    )
    
    if len(regions_a_comparer) < 2:
        st.warning("Veuillez sélectionner au moins 2 régions pour la comparaison")
    else:
        # Filtrer les données pour les régions sélectionnées
        compare_df = filtered_df[filtered_df['region'].isin(regions_a_comparer)]
        
        # Agrégation par région et date
        region_daily_compare = compare_df.groupby(['Date', 'region']).agg({
            'stock_actuel': 'sum',
            'capacite_totale_actuelle': 'sum',
            'apports_journaliers': 'sum',
            'lachers_journaliers': 'sum'
        }).reset_index()
        
        region_daily_compare['taux_remplissage'] = (region_daily_compare['stock_actuel'] / region_daily_compare['capacite_totale_actuelle']) * 100
        
        # Comparaison des taux de remplissage
        st.subheader("Comparaison des taux de remplissage")
        
        fig_compare_fill = px.line(
            region_daily_compare,
            x='Date',
            y='taux_remplissage',
            color='region',
            labels={'taux_remplissage': 'Taux de remplissage (%)', 'Date': 'Date', 'region': 'Région'},
            title="Comparaison des taux de remplissage par région"
        )
        
        fig_compare_fill.update_layout(height=500, hovermode="x unified")
        
        st.plotly_chart(fig_compare_fill, use_container_width=True)
        
        # Comparaison des apports journaliers
        st.subheader("Comparaison des apports journaliers")
        
        fig_compare_inflow = px.line(
            region_daily_compare,
            x='Date',
            y='apports_journaliers',
            color='region',
            labels={'apports_journaliers': 'Apports journaliers (Mm³)', 'Date': 'Date', 'region': 'Région'},
            title="Comparaison des apports journaliers par région"
        )
        
        fig_compare_inflow.update_layout(height=500, hovermode="x unified")
        
        st.plotly_chart(fig_compare_inflow, use_container_width=True)
        
        # Radar chart pour comparaison multi-critères
        st.subheader("Comparaison multi-critères (données récentes)")
        
        # Prendre les données les plus récentes pour chaque région
        latest_date = filtered_df['Date'].max()
        latest_region_compare = compare_df[compare_df['Date'] == latest_date]
        
        # Agréger par région pour le radar chart
        radar_data = latest_region_compare.groupby('region').agg({
            'taux_remplissage': 'mean',
            'stock_actuel': 'sum',
            'capacite_totale_actuelle': 'sum',
            'apports_journaliers': 'sum',
            'lachers_journaliers': 'sum'
        }).reset_index()
        
        # Normaliser les données pour le radar chart
        for col in ['stock_actuel', 'capacite_totale_actuelle', 'apports_journaliers', 'lachers_journaliers']:
            max_val = radar_data[col].max()
            if max_val > 0:  # Éviter la division par zéro
                radar_data[f'{col}_norm'] = (radar_data[col] / max_val) * 100
            else:
                radar_data[f'{col}_norm'] = 0
        
        # Créer le radar chart
        fig_radar = go.Figure()
        
        categories = [
            'Taux de remplissage (%)', 
            'Stock actuel (normalisé)', 
            'Capacité totale (normalisée)',
            'Apports (normalisés)',
            'Lâchers (normalisés)'
        ]
        
        for region in radar_data['region']:
            region_data = radar_data[radar_data['region'] == region]
            fig_radar.add_trace(go.Scatterpolar(
                r=[
                    region_data['taux_remplissage'].values[0],
                    region_data['stock_actuel_norm'].values[0],
                    region_data['capacite_totale_actuelle_norm'].values[0],
                    region_data['apports_journaliers_norm'].values[0],
                    region_data['lachers_journaliers_norm'].values[0]
                ],
                theta=categories,
                fill='toself',
                name=region
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=500,
            title="Comparaison multi-critères entre régions (données normalisées)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Comparaison du nombre de barrages et capacité totale
        st.subheader("Comparaison de la capacité et du nombre de barrages")
        
        # Calculer le nombre de barrages par région
        barrage_count_compare = latest_region_compare.groupby('region')['nom_barrage'].nunique().reset_index()
        barrage_count_compare.columns = ['region', 'nombre_barrages']
        
        # Fusionner avec les données de capacité
        capacity_compare = latest_region_compare.groupby('region')['capacite_totale_actuelle'].sum().reset_index()
        comparison_data = pd.merge(barrage_count_compare, capacity_compare, on='region')
        
        # Créer un graphique à barres groupées
        fig_barrage_capacity = go.Figure()
        
        fig_barrage_capacity.add_trace(go.Bar(
            x=comparison_data['region'],
            y=comparison_data['nombre_barrages'],
            name="Nombre de barrages",
            marker_color='lightblue',
            text=comparison_data['nombre_barrages'],
            textposition='auto'
        ))
        
        fig_barrage_capacity.add_trace(go.Bar(
            x=comparison_data['region'],
            y=comparison_data['capacite_totale_actuelle'],
            name="Capacité totale (Mm³)",
            marker_color='lightgreen',
            text=comparison_data['capacite_totale_actuelle'].round(2),
            textposition='auto'
        ))
        
        fig_barrage_capacity.update_layout(
            barmode='group',
            title="Comparaison du nombre de barrages et de la capacité totale par région",
            xaxis_title="Région",
            height=500
        )
        
        st.plotly_chart(fig_barrage_capacity, use_container_width=True)

with tab4:
    st.header("Données brutes")
    
    # Options de filtrage pour les données brutes
    col1, col2 = st.columns(2)
    
    with col1:
        show_all_columns = st.checkbox("Afficher toutes les colonnes", False)
    
    with col2:
        sort_by = st.selectbox(
            "Trier par",
            options=["Date", "region", "nom_barrage", "taux_remplissage"],
            index=0
        )
    
    # Préparation des données à afficher
    if show_all_columns:
        display_df = filtered_df.sort_values(by=sort_by)
    else:
        # Sélection des colonnes les plus importantes
        display_df = filtered_df[[
            'Date', 'region', 'nom_barrage', 'stock_actuel', 
            'capacite_totale_actuelle', 'taux_remplissage', 
            'apports_journaliers', 'lachers_journaliers'
        ]].sort_values(by=sort_by)
    
    # Afficher les données filtrées
    st.dataframe(display_df, use_container_width=True)
    
    # Option pour télécharger les données filtrées
    csv = filtered_df.to_csv(index=False)
    
    # Générer un nom de fichier basé sur les régions sélectionnées
    if len(selected_regions) <= 3:
        regions_str = "_".join(selected_regions)
    else:
        regions_str = f"{len(selected_regions)}_regions"
    
    st.download_button(
        label="Télécharger les données filtrées (CSV)",
        data=csv,
        file_name=f"barrages_{regions_str}_{start_date.date()}_{end_date.date()}.csv",
        mime="text/csv",
    )

# Ajouter une section d'aide
with st.sidebar.expander("Aide"):
    st.markdown("""
    ## Comment utiliser ce dashboard

    1. **Filtrage par région**: Dans la barre latérale, sélectionnez les régions qui vous intéressent.
    2. **Navigation**: Utilisez les onglets pour explorer différentes analyses:
        - **Vue d'ensemble**: Statistiques globales et graphiques pour toutes les régions sélectionnées
        - **Analyse par région**: Analyse détaillée d'une région spécifique
        - **Comparaison**: Comparez plusieurs régions entre elles
        - **Données brutes**: Consultez et téléchargez les données filtrées
        - **Carte** : Visualisation géolocalisée des enjeux
    3. **Filtrage par date (optionnel)**: Activez cette option pour analyser une période spécifique
    
    Pour toute question ou assistance, contactez l'administrateur du système.
    """)

# Footer
st.markdown("---")
st.markdown("Dashboard créé avec Streamlit pour l'analyse des données de barrages par région")

# Ajout d'informations complémentaires
st.sidebar.markdown("---")
st.sidebar.markdown("### Informations")

# Ajout d'une section de statistiques rapides dans la sidebar
with st.sidebar.expander("Statistiques rapides"):
    if not filtered_df.empty:
        # Dernière date de mise à jour
        last_update = filtered_df['Date'].max().strftime('%d/%m/%Y')
        st.write(f"📅 Dernière mise à jour: **{last_update}**")
        
        # Nombre total de barrages
        total_barrages = filtered_df[filtered_df['Date'] == filtered_df['Date'].max()]['nom_barrage'].nunique()
        st.write(f"🏞️ Nombre total de barrages: **{total_barrages}**")
        
        # Région avec le meilleur taux de remplissage
        latest_data = filtered_df[filtered_df['Date'] == filtered_df['Date'].max()]
        region_fill_rates = latest_data.groupby('region').apply(
            lambda x: (x['stock_actuel'].sum() / x['capacite_totale_actuelle'].sum()) * 100 if x['capacite_totale_actuelle'].sum() > 0 else 0
        ).reset_index()
        region_fill_rates.columns = ['region', 'taux_remplissage']
        
        if not region_fill_rates.empty:
            best_region = region_fill_rates.loc[region_fill_rates['taux_remplissage'].idxmax()]
            st.write(f"🥇 Meilleur taux de remplissage: **{best_region['region']}** ({best_region['taux_remplissage']:.2f}%)")
            
            worst_region = region_fill_rates.loc[region_fill_rates['taux_remplissage'].idxmin()]
            st.write(f"⚠️ Plus faible taux: **{worst_region['region']}** ({worst_region['taux_remplissage']:.2f}%)")

# Ajout d'une section de recherche de barrage spécifique
with st.sidebar.expander("Rechercher un barrage"):
    barrage_search = st.text_input("Nom du barrage", "")
    
    if barrage_search:
        # Recherche insensible à la casse et partielle
        matching_barrages = df[df['nom_barrage'].str.contains(barrage_search, case=False)]
        
        if not matching_barrages.empty:
            unique_matches = matching_barrages['nom_barrage'].unique()
            st.write(f"**{len(unique_matches)} barrage(s) trouvé(s):**")
            for i, barrage in enumerate(unique_matches):
                region = df[df['nom_barrage'] == barrage]['region'].iloc[0]
                st.write(f"{i+1}. {barrage} ({region})")
        else:
            st.write("Aucun barrage trouvé.")

# Ajout d'une fonction d'exportation avancée
with st.sidebar.expander("Exportation avancée"):
    export_format = st.radio(
        "Format d'exportation",
        options=["CSV", "Excel"],
        index=0
    )
    
    export_content = st.multiselect(
        "Contenu à exporter",
        options=["Données brutes", "Statistiques par région", "Statistiques par barrage"],
        default=["Données brutes"]
    )
    
    if st.button("Préparer l'exportation"):
        # Création d'un dictionnaire pour stocker les différents DataFrames
        export_data = {}
        
        if "Données brutes" in export_content:
            export_data["Données brutes"] = filtered_df
        
        if "Statistiques par région" in export_content:
            # Agrégation par région pour la dernière date disponible
            latest_data = filtered_df[filtered_df['Date'] == filtered_df['Date'].max()]
            region_stats = latest_data.groupby('region').agg({
                'nom_barrage': 'nunique',
                'stock_actuel': 'sum',
                'capacite_totale_actuelle': 'sum',
                'apports_journaliers': 'sum',
                'lachers_journaliers': 'sum'
            }).reset_index()
            
            region_stats['taux_remplissage'] = (region_stats['stock_actuel'] / region_stats['capacite_totale_actuelle']) * 100
            region_stats.rename(columns={'nom_barrage': 'nombre_barrages'}, inplace=True)
            
            export_data["Statistiques par région"] = region_stats
        
        if "Statistiques par barrage" in export_content:
            # Agrégation par barrage pour la dernière date disponible
            barrage_stats = latest_data.groupby(['region', 'nom_barrage']).agg({
                'stock_actuel': 'sum',
                'capacite_totale_actuelle': 'sum',
                'apports_journaliers': 'sum',
                'lachers_journaliers': 'sum',
                'taux_remplissage': 'mean'
            }).reset_index()
            
            export_data["Statistiques par barrage"] = barrage_stats
        
        # Générer le nom du fichier
        if len(selected_regions) <= 3:
            regions_str = "_".join(selected_regions)
        else:
            regions_str = f"{len(selected_regions)}_regions"
        
        filename = f"barrages_{regions_str}_{datetime.datetime.now().strftime('%Y%m%d')}"
        
        # Préparer le fichier d'exportation selon le format choisi
        if export_format == "CSV":
            # Pour CSV, on crée un fichier par DataFrame
            for name, df_to_export in export_data.items():
                csv_data = df_to_export.to_csv(index=False)
                safe_name = name.replace(" ", "_").lower()
                st.download_button(
                    label=f"Télécharger {name}",
                    data=csv_data,
                    file_name=f"{filename}_{safe_name}.csv",
                    mime="text/csv",
                )
        else:  # Excel
            # Pour Excel, on peut mettre plusieurs DataFrames dans différentes feuilles
            # Note: Cette partie nécessite que le package openpyxl soit installé
            try:
                import io
                from io import BytesIO
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for name, df_to_export in export_data.items():
                        safe_name = name.replace(" ", "_").lower()[:31]  # Excel a une limite de 31 caractères pour les noms de feuilles
                        df_to_export.to_excel(writer, sheet_name=safe_name, index=False)
                
                # Préparer le fichier pour téléchargement
                output.seek(0)
                excel_data = output.read()
                
                st.download_button(
                    label="Télécharger toutes les données (Excel)",
                    data=excel_data,
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                
            except ImportError:
                st.error("L'exportation Excel nécessite le package openpyxl. Veuillez l'installer ou choisir le format CSV.")

# Ajout d'une fonctionnalité pour visualiser les tendances
if 'tab1' in locals():
    with tab1:
        st.markdown("---")
        st.subheader("Analyse des tendances")
        
        # Calcul des tendances sur les derniers jours
        trend_days = st.slider("Nombre de jours pour l'analyse des tendances", 7, 90, 30)
        
        # S'assurer que nous avons suffisamment de données
        if filtered_df['Date'].nunique() >= trend_days:
            # Obtenir les dates pour la période d'analyse
            all_dates = sorted(filtered_df['Date'].unique())
            if len(all_dates) > trend_days:
                trend_dates = all_dates[-trend_days:]
                trend_df = filtered_df[filtered_df['Date'].isin(trend_dates)]
            else:
                trend_df = filtered_df
            
            # Calculer les tendances par région
            first_day = trend_df.groupby(['region', 'Date']).agg({
                'taux_remplissage': 'mean'
            }).reset_index()
            
            # Obtenir la première et la dernière date pour chaque région
            first_date = trend_dates[0]
            last_date = trend_dates[-1]
            
            first_values = first_day[first_day['Date'] == first_date].set_index('region')
            last_values = first_day[first_day['Date'] == last_date].set_index('region')
            
            # Calculer les variations
            trend_results = []
            
            for region in first_values.index.intersection(last_values.index):
                first_val = first_values.loc[region, 'taux_remplissage']
                last_val = last_values.loc[region, 'taux_remplissage']
                change = last_val - first_val
                percent_change = (change / first_val * 100) if first_val > 0 else 0
                
                trend_results.append({
                    'region': region,
                    'taux_initial': first_val,
                    'taux_final': last_val,
                    'variation_absolue': change,
                    'variation_pct': percent_change
                })
            
            if trend_results:
                trend_df = pd.DataFrame(trend_results).sort_values('variation_pct', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique de variation en pourcentage
                    fig_trend_pct = px.bar(
                        trend_df,
                        x='region',
                        y='variation_pct',
                        color='variation_pct',
                        color_continuous_scale=px.colors.diverging.RdBu,
                        color_continuous_midpoint=0,
                        labels={'variation_pct': 'Variation (%)', 'region': 'Région'},
                        title=f"Variation du taux de remplissage sur {trend_days} jours (%)"
                    )
                    
                    fig_trend_pct.update_layout(height=400)
                    fig_trend_pct.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
                    
                    st.plotly_chart(fig_trend_pct, use_container_width=True)
                
                with col2:
                    # Graphique des taux initiaux et finaux
                    trend_compare = pd.melt(
                        trend_df, 
                        id_vars=['region'], 
                        value_vars=['taux_initial', 'taux_final'],
                        var_name='période', 
                        value_name='taux'
                    )
                    
                    # Renommer les périodes pour l'affichage
                    trend_compare['période'] = trend_compare['période'].map({
                        'taux_initial': f'Début ({first_date.strftime("%d/%m/%Y")})',
                        'taux_final': f'Fin ({last_date.strftime("%d/%m/%Y")})'
                    })
                    
                    fig_trend_compare = px.bar(
                        trend_compare,
                        x='region',
                        y='taux',
                        color='période',
                        barmode='group',
                        labels={'taux': 'Taux de remplissage (%)', 'region': 'Région', 'période': 'Période'},
                        title="Comparaison des taux de remplissage début vs fin de période"
                    )
                    
                    fig_trend_compare.update_layout(height=400)
                    st.plotly_chart(fig_trend_compare, use_container_width=True)
            else:
                st.warning(f"Données insuffisantes pour calculer les tendances sur {trend_days} jours.")
        else:
            st.warning(f"Données insuffisantes pour calculer les tendances sur {trend_days} jours.")


# In[ ]:




