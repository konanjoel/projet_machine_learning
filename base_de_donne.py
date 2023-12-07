
import sqlite3
import pandas as pd
import wget
import zipfile
import os

url = "https://github.com/eishkina-estia/ML2023/raw/main/data/New_York_City_Taxi_Trip_Duration.zip"
destination = os.path.dirname(__file__)
fichier_zip = wget.download(url, out=destination)

with zipfile.ZipFile(fichier_zip, 'r') as zip_ref:
        zip_ref.extractall(destination)

os.remove(fichier_zip)
fichier_csv = os.path.join(destination, "New_York_City_Taxi_Trip_Duration.csv")
data = pd.read_csv(fichier_csv)
#data = pd.read_csv('New_York_City_Taxi_Trip_Duration.csv')
# Création ou connexion à la base de données

nom_base_de_donnees = 'ma_base_de_donnees.db'
connexion = sqlite3.connect(nom_base_de_donnees)

# Insertion du DataFrame dans la base de données
data.to_sql(name='ma_table', con=connexion, if_exists='replace', index=False)

# Fermeture de la connexion
connexion.close()

print(f"DataFrame inséré dans la base de données: {nom_base_de_donnees}")