
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42




def preprocessing(data):
    # check if there are any columns containing unique values for each row. If so, drop them
    data = data.drop(columns=['id'])
    # dropoff_datetime variable is added only to train data and thus cannot be used by the predictive model. Drop this feature
    data = data.drop(columns=['dropoff_datetime'])
    # pickup_datetime contains date and time when the meter was engaged. Check the type of this feature and change it to datetime if it is another type.
    print(data['pickup_datetime'])
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    # split data into train and test: 30% of data for test.
    from sklearn.model_selection import train_test_split

    X = data.drop(columns=['trip_duration'])
    y = data['trip_duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    # target transformation
    def transform_target(y):
        return np.log1p(y).rename('log_'+y.name)

    y = transform_target(y)
    y_train = transform_target(y_train)
    y_test = transform_target(y_test) 
    
    from sklearn.metrics import mean_squared_error

    y_baseline = y_train.mean()
    print(f'Baseline prediction: {y_baseline:.2f} (transformed)')
    # np.expm1 is the inverse of log1p
    print(f'Baseline prediction: {np.expm1(y_baseline):.0f} (seconds)')

    print(f'RMSLE on train data: {mean_squared_error([y_baseline]*len(y_train), y_train, squared=False):.3f}')
    print(f'RMSLE on train data: {mean_squared_error([y_baseline]*len(y_test), y_test, squared=False):.3f}')   
    # Number of trips ~ date  
    X['pickup_date'] = X['pickup_datetime'].dt.date 
    df_abnormal_dates = X.groupby('pickup_date').size()
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < df_abnormal_dates.quantile(0.02)]
    print(abnormal_dates)
    # New features
    def step1_add_features(X):
        res = X.copy()
        res['weekday'] = res['pickup_datetime'].dt.weekday
        res['month'] = res['pickup_datetime'].dt.month
        res['hour'] = res['pickup_datetime'].dt.hour
        res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
        return res

    X = step1_add_features(X)
    X_train = step1_add_features(X_train)
    X_test = step1_add_features(X_test)
    return X,X_train,X_test,y_train, y_test,y


      




