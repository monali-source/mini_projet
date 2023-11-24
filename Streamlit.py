import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

import folium
from folium.plugins import HeatMap
from pathlib import Path
import geopandas as gpd


st.title("Musique générale 1950 à 2020")

#st.header('1°/ Mes données')

#import csv
@st.cache_data
def load_data():
   data = "Projet_Data_Music-main/data/tcc_ceds_music.csv"
   data = pd.read_csv(data)
   return data

df = load_data()
st.header('Exploration des données')
st.write(df)





#details sur artistes
artists = df['artist_name'].unique()
artists=['Toutes'] + list(artists)

years = df['release_date'].unique()
years = ['Toutes'] + list(years)  # Ajoutez une option "Toutes" pour toutes les années


#Sélectionnez l'artists
selected_artist = st.selectbox("Sélectionnez une artists", artists)

#Sélectionnez l'année
selected_year = st.selectbox("Sélectionnez une année", years)

#Filtrage des données en fonction des sélectionsù
filtered_data = df.copy()

if selected_artist != 'Toutes':
    filtered_data = filtered_data[filtered_data['artist_name'] == selected_artist]

if selected_year != 'Toutes':
    filtered_data = filtered_data[filtered_data['release_date'] == selected_year]


#Afficher les données filtrées
st.write(f"Données pour la région {selected_artist}:")
st.write(filtered_data)





#details sur artistes
artists_nam = df['artist_name']
artists_nam =['Toutes'] + list(artists_nam)

topics = df['topic'].unique()
topics = ['Toutes'] + list(topics)

artists = df["artist_name"].value_counts()[:20].sort_values(ascending = True) 

#Sélectionnez l'artists
selected_artist_nam = st.selectbox("Sélectionnez une artists", artists_nam)

#Sélectionnez l'année
selected_topics = st.selectbox("Sélectionnez une topics", topics)



filtered_data = df.copy()

if selected_artist_nam != 'Toutes':
    filtered_data = filtered_data[filtered_data['artist_name'] == selected_artist_nam]

if selected_topics != 'Toutes':
    filtered_data = filtered_data[filtered_data['topic'] == selected_topics]

# Afficher les données filtrées
st.write(f"Données pour la région {selected_artist_nam}:")
st.write(filtered_data)








data_load_state = st.text('Chargement...')

data = pd.read_csv("Projet_Data_Music-main/Data/top10s.csv", encoding = "ISO-8859-1")
st.write(data.head(20))

data_load_state.text('Chargement...réussi!')






st.write("Pour mieux voir les résultats, on les représente avec des couleurs. On en profite aussi pour séparer les colonnes entre kes numériques et catégoriques pour pouvoir executer cette commande.")

fig, ax = plt.subplots()

categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(exclude=['object']).columns
nrows = data.shape[0]
ncols = data.shape[1]
sns.heatmap(data[numerical_cols].corr(), vmin = -1, vmax = 1, ax=ax)

st.write(fig)

st.write("Enfin, on ajoute les valeurs de la table de correlation pour plus de précision")
plt.clf()

fig, ax = plt.subplots()
plt.figure(figsize=(7, 6))
sns.heatmap(data[numerical_cols].corr(), annot = True, fmt = '.2f', cmap='Blues', vmin = -1, vmax = 1, ax=ax)
plt.title('Correlation entre les colonnes dans la table Spotify')

st.write(fig)


st.write("On remarque que les colonnes des paramètres ont donnés ces résultats:")

st.write(" - Acous lié à nrgy (-0.56)")

st.write(" - Val lié à dnce (0.50)")

st.write(" - dB lié à nrgy (0.54)")

st.write("On a ensuite représenté graphiquement ces paramètres entre eux:")

plt.clf()
fig, ax = plt.subplots()
data.plot(x='acous',y='nrgy',kind='scatter', title='Relation entre Energy et Acousticness',color='r', ax=ax)
plt.xlabel('Acousticness')
plt.ylabel('Energy')

st.write(fig)

plt.clf()
fig, ax = plt.subplots()
data.plot(x='nrgy',y='dB',kind='scatter', title='Relation entre Loudness (dB) et Energy',color='b', ax=ax)
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
st.write(fig)

plt.clf()
fig, ax = plt.subplots()
data.plot(x='val',y='dnce',kind='scatter', title='Relation entre Loudness (dB) et Valence',color='g', ax=ax)
plt.xlabel('Valence')
plt.ylabel('Loudness (dB)')
st.write(fig)

st.write("On peut maintenant répondre à des problématiques intéressantes par rapport aux données trouvées.")

st.write("Par example, on peut trouver qui sont les artistes ayant le plus de chansons enregistrés dans la table de données")

artists = data['artist'].unique()
st.write("Il y a 184 artistes : ", len(artists))

artists = data['artist'].value_counts().reset_index().head(10)
st.write(artists)


st.write("Au vu des résultats de la commande précédente, remarque que Katy Perry est l'artiste avec le plus de titres enregistrés dans le top 10 de Spotify entre 2010 et 2019")


st.write("Ensuitetitres réapparaissent les affiche graphiquement :")


plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(15,10))
sns.countplot(y=data.title, order=pd.value_counts(data.title).iloc[:19].index, data=data, ax=ax)
topMusics = data['title'].value_counts().head(19).index
plt.title("Titres apparaissant plus d'une fois")
st.write(fig)


st.write("On cherche ensuite la répartition de ces titres durant les années :")

plt.clf()
fig, ax = plt.subplots()
plt.figure(figsize=(20,10))
for i in topMusics:
  tmp = []
  for y in range(2010,2020):
    songs = data[data['year'] == y][data['title'] == i]
    tmp.append(songs.shape[0])
  sns.lineplot(x=list(range(2010,2020)),y=tmp, ax=ax)
plt.legend(list(topMusics))
plt.title("Evolution de chaque titre du top 10 répété plus d'une fois à travers les années")
st.write(fig)


st.write("On remarque que le titre 'Sugar' de Maroon 5 est revenu 2 fois durant la même année")

st.write(data[data['title']== 'Sugar'])

genres = data['top_genre'].value_counts().reset_index().head(10)

# Reset DataFrame
genres_reset_index = genres.reset_index()

plt.figure(figsize=(23, 10))
sns.barplot(x='index', y='top_genre', data=genres_reset_index)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Top 10 Genres')

plt.show()

st.write("On souhaite voir d'où viennent les artistes les plus populaires dans le top 10 des artistes de Spotify")

st.write(artists)

st.write("Vu que cette donnée ne vient pas de notre table, on entre les données manuellement")

dicArtists = {
    'Katy Perry':"Santa Barbara",
    'Justin Bieber':"London Canada",
     'Rihanna':"Saint Michael",
    'Maroon 5':"Los Angeles",
    'Lady Gaga':"Manhattan",
    'Bruno Mars':"Honolulu", 
    'The Chainsmokers':"Times Square" ,
    'Pitbull':"Miami",
    'Shawn Mendes':"Toronto",
    'Ed Sheeran':"United Kingdom", 
  }

st.write(dicArtists)

st.write("On définit les coordonnées en longitude et lattitude")

import geocoder
listGeo = []

for value in (dicArtists.values()):
    g = geocoder.arcgis(value)
    listGeo.append(g.latlng)

st.write(listGeo)

top_genres =[]
for key in (dicArtists.keys()):
    top_genres.append(data[data['artist']== key].top_genre.unique())

st.write(top_genres)

lat = []
log = []
for i in listGeo:
    lat.append(i[0])
    log.append(i[1])


colors = {
 'dance pop': 'pink',
 'pop': 'blue',
 'barbadian pop': 'green',
 'electropop': 'orange',
 'canadian pop': 'red',
}

st.write("Après les avoir définit, voila les coordonnées des artistes sélectionnés :")

dfLocation = pd.DataFrame(columns=['Name','Lat','Log','Gen'])
dfLocation['Name'] = list(dicArtists.keys())
dfLocation['Gen']  = np.array(top_genres)
dfLocation['Lat']  = lat
dfLocation['Log']  = log


st.write(dfLocation)



