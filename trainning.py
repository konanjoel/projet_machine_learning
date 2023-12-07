import sqlite3
import pandas as pd
from preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Nom de la base de données
nom_base_de_donnees = 'ma_base_de_donnees.db'

# Connexion à la base de données
connexion = sqlite3.connect(nom_base_de_donnees)

# Requête SQL pour récupérer les données de la table (remplacez 'ma_table' par le nom de votre table)
requete_sql = "SELECT * FROM ma_table"

# Utilisation de pandas pour lire les données de la base de données dans un DataFrame
data = pd.read_sql(requete_sql, connexion)

# Fermeture de la connexion
connexion.close()


X,X_train,X_test,y_train, y_test,y =preprocessing(data)

num_features = ['abnormal_period', 'hour']
cat_features = ['weekday', 'month']
train_features = num_features + cat_features

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
    ('scaling', StandardScaler(), num_features)]
)

pipeline = Pipeline(steps=[
    ('ohe_and_scaling', column_transformer),
    ('regression', Ridge())
])

model = pipeline.fit(X_train[train_features], y_train)
y_pred_train = model.predict(X_train[train_features])
y_pred_test = model.predict(X_test[train_features])

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test, squared=False))
print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))
# Folium
import folium

#Distance estimation
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

distance_haversine = haversine_array(X.pickup_latitude, X.pickup_longitude, X.dropoff_latitude, X.dropoff_longitude)
log_distance_haversine = np.log1p(distance_haversine)
print(f'Correlation between target and distance_haversine: {np.corrcoef(x=distance_haversine,y=y)[0,1]:.2f}')
print(f'Correlation between target and log_distance_haversine: {np.corrcoef(x=log_distance_haversine,y=y)[0,1]:.2f}')

# Outliers
def is_rare_point(X, latitude_column, longitude_column, qmin_lat, qmax_lat, qmin_lon, qmax_lon):
  lat_min = X[latitude_column].quantile(qmin_lat)
  lat_max = X[latitude_column].quantile(qmax_lat)
  lon_min = X[longitude_column].quantile(qmin_lon)
  lon_max = X[longitude_column].quantile(qmax_lon)

  res = (X[latitude_column] < lat_min) | (X[latitude_column] > lat_max) | \
        (X[longitude_column] < lon_min) | (X[longitude_column] > lon_max)
  return res
latitude_column, longitude_column = "pickup_latitude", "pickup_longitude"
#fonction map 
def show_circles_on_map(data, latitude_column, longitude_column, color):
    """
    The function draws map with circles on it.
    The center of the map is the mean of coordinates passed in data.

    data: DataFrame that contains columns latitude_column and longitude_column
    latitude_column: string, the name of column for latitude coordinates
    longitude_column: string, the name of column for longitude coordinates
    color: string, the color of circles to be drawn
    """

    location = (data[latitude_column].mean(), data[longitude_column].mean())
    m = folium.Map(location=location)

    for _, row in data.iterrows():
        folium.Circle(
            radius=100,
            location=(row[latitude_column], row[longitude_column]),
            color=color,
            fill_color=color,
            fill=True
        ).add_to(m)

    return m

is_rare_point_vec = is_rare_point(X, latitude_column, longitude_column, 0.01, 0.995, 0, 0.95)
m = show_circles_on_map(data[~is_rare_point_vec].sample(1000), latitude_column, longitude_column, "blue")

rare_points = X[[latitude_column, longitude_column]][is_rare_point_vec]
color_rare = 'red'

for _, row in rare_points.sample(100).iterrows():
    folium.Circle(
        radius=100,
        location=(row[latitude_column], row[longitude_column]),
        color=color_rare,
        fill_color=color_rare,
        fill=True
    ).add_to(m)
m

# Check new features
def is_high_traffic_trip(X):
  return ((X['hour'] >= 8) & (X['hour'] <= 19) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
         ((X['hour'] >= 13) & (X['hour'] <= 20) & (X['weekday'] == 5))

def is_high_speed_trip(X):
  return ((X['hour'] >= 2) & (X['hour'] <= 5) & (X['weekday'] >= 0) & (X['weekday'] <= 4)) | \
         ((X['hour'] >= 4) & (X['hour'] <= 7) & (X['weekday'] >= 5) & (X['weekday'] <= 6))
def step2_add_features(X):
  res = X.copy()
  distance_haversine = haversine_array(res.pickup_latitude, res.pickup_longitude, res.dropoff_latitude, res.dropoff_longitude)
  res['log_distance_haversine'] = np.log1p(distance_haversine)
  res['is_high_traffic_trip'] = is_high_traffic_trip(X).astype(int)
  res['is_high_speed_trip'] = is_high_traffic_trip(X).astype(int)
  res['is_rare_pickup_point'] = is_rare_point(X, "pickup_latitude", "pickup_longitude", 0.01, 0.995, 0, 0.95).astype(int)
  res['is_rare_dropoff_point'] = is_rare_point(X, "dropoff_latitude", "dropoff_longitude", 0.01, 0.995, 0.005, 0.95).astype(int)

  return res

X = step2_add_features(X)
X_train = step2_add_features(X_train)
X_test = step2_add_features(X_test)

num_features = ['log_distance_haversine', 'hour',
                'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                'is_rare_pickup_point', 'is_rare_dropoff_point']
cat_features = ['weekday', 'month']
train_features = num_features + cat_features

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
    ('scaling', StandardScaler(), num_features)]
)

pipeline = Pipeline(steps=[
    ('ohe_and_scaling', column_transformer),
    ('regression', Ridge())
])

model = pipeline.fit(X_train[train_features], y_train)
y_pred_train = model.predict(X_train[train_features])
y_pred_test = model.predict(X_test[train_features])

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test, squared=False))
print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))
# order features

def step3_process_features(X):
  res = X.copy()
  res['vendor_id'] = res['vendor_id'].map({1:0, 2:1})
  res['store_and_fwd_flag'] = res['store_and_fwd_flag'].map({'N':0, 'Y':1})
  res.loc[res['passenger_count'] > 6, 'passenger_count'] = 6

  return res

X = step3_process_features(X)
X_train = step3_process_features(X_train)
X_test = step3_process_features(X_test)

num_features = ['log_distance_haversine', 'hour',
                'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                'is_rare_pickup_point', 'is_rare_dropoff_point',
                'vendor_id', 'store_and_fwd_flag']
cat_features = ['weekday', 'month']
train_features = num_features + cat_features

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
    ('scaling', StandardScaler(), num_features)]
)

pipeline = Pipeline(steps=[
    ('ohe_and_scaling', column_transformer),
    ('regression', Ridge())
])

model = pipeline.fit(X_train[train_features], y_train)
y_pred_train = model.predict(X_train[train_features])
y_pred_test = model.predict(X_test[train_features])

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test, squared=False))
print("Test R2 = %.4f" % r2_score(y_test, y_pred_test))

