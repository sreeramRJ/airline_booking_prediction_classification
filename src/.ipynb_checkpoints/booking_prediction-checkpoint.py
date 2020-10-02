import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
from numpy import mean

import sys

# function to read inputs and merge
def data_read(x,y):
    iata = pd.read_csv(x)
    events = pd.read_csv(y)
    booking_data = events.merge(iata,left_on='origin',right_on='iata_code').drop('iata_code',axis=1)
    booking_data = booking_data.rename(columns={'lat':'origin_lat','lon':'origin_lon'})
    booking_data = booking_data.merge(iata,left_on='destination',right_on='iata_code').drop('iata_code',axis=1)
    booking_data = booking_data.rename(columns={'lat':'destination_lat','lon':'destination_lon'})
    return booking_data
	
#function to handle data pre-processing    
def data_preprocessing(booking_data):
    #remove NA values
	booking_data = booking_data.dropna()
	
	booking_data = add_columns(booking_data)
	
	booking_data = encoding_function(booking_data)
	
	# Removing unwanted columns
	booking_data.drop(['date_from','date_to','look_on','origin','destination','event_type','ts','user_id','origin_lat','origin_lon','destination_lat','destination_lon'],axis=1,inplace=True)
	
	#remove records where 'days_before_plan' is negative as it is meaningless
	booking_data = booking_data.loc[booking_data['days_before_plan'] > 0]

    return booking_data
    
#function to calculate the geo-distance between 2 lat-lon co-ordinates
def haversine_vectorize(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula ))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return km
	
# This method is responsible to calculate the number of days between 2 dates.
def no_of_days(date_from,looking_on):
    date_format = "%Y-%m-%d"
    a = datetime.strptime(looking_on, date_format)
    b = datetime.strptime(date_from, date_format)
    delta = b - a #days
    return delta.days
	
#Add columns 'days_before_plan', 'trip_duration', 'activity_month', 'activity_hour' and 'travel_start_month'
def add_columns(df):
	#calculate geo_distance
	df['geo_distance(km)'] = haversine_vectorize(df['origin_lon'],df['origin_lat'],df['destination_lon'],df['destination_lat'])
	# This feature 'days_before_plan' will be used to calculate the no of days before the user is planning to book/search for a flight.
	df['look_on']=df['ts'].str.split(" ", n = 1, expand = True)[0]        
	df['days_before_plan'] = df.apply(lambda row : no_of_days(row['date_from'], row['look_on']), axis = 1)
	# This feature 'trip_duration' represents for how many days the trip was planned for
	df['trip_duration'] = df.apply(lambda row : no_of_days(row['date_to'], row['date_from']), axis = 1)
	#Extract search/book activity month and hour, travel start date month as these features might give an indication of seasonality to the model
	df['activity_month'] = pd.to_datetime(df['ts']).dt.month
	df['travel_start_month'] = pd.to_datetime(df['date_from']).dt.month
	df['activity_hour'] = pd.to_datetime(df['ts']).dt.hour
	return df
	
def encoding_function(df):
	# Encoding origin column
	le = LabelEncoder()
	le.fit(df['origin'])
	origin_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
	df['origin_enc'] = le.transform(df['origin'])

	#Encoding destination column
	le = LabelEncoder()
	le.fit(df['destination'])
	destination_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
	df['destination_enc'] = le.transform(df['destination'])
			
	# Encoding our label column i.e 'event_type'
	le = LabelEncoder()
	le.fit(df['event_type'])
	event_type_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
	df['event_type_enc'] = le.transform(df['event_type'])
	return df

def sampling_func(X_train,y_train):
	# Handling unbalanced dataset with SMOTE oversampling technique
	sm = SMOTE(random_state=12)
	x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
	return x_train_res, y_train_res
	
    
def model_evaluation(model,X,y):
    lg = lgb.LGBMClassifier(silent=False)
    param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }
    grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
    grid_search.fit(X,y)
    return grid_search.best_estimator_
    
def model_fit(x_train,x_test,y_train,y_test):
    # fit a lightGBM classifier model to the data
    # Why LightGBM? - It works great with categorical features
    # categorical_feature = ['activity_month','origin_enc','destination_enc','travel_start_month','activity_hour']
    params = {'verbose': -1}
    model = lgb.LGBMClassifier()
    model2 = model_evaluation(model,x_train,y_train)
    model2.fit(x_train, y_train, eval_set=(x_test, y_test), feature_name='auto', categorical_feature = ['activity_month','origin_enc','destination_enc','travel_start_month','activity_hour'], early_stopping_rounds=100)
    print(); print(model2)
    # make predictions
    y_pred = model2.predict(X_test)
    # importance of each attribute
    model2.booster_.feature_importance()
    fea_imp_ = pd.DataFrame({'cols':x_train.columns, 'fea_imp':model.feature_importances_})
    return y_pred, fea_imp_
	
if __name__=="__main__":
	warnings.filterwarnings('ignore')
	iata = sys.argv[1]
	events = sys.argv[2]
	#read data
	booking_data = data_read(iata,events)
	
	#clean-up and pre-process the dataset to be suitable for model training and evaluation
    booking_data_preprocessed = data_preprocessing(booking_data)
    
	#split the dataset into independant and dependant features for model training and validation
	X = booking_data_preprocessed.loc[:, booking_data_preprocessed.columns != 'event_type_enc']
	y = booking_data_preprocessed.loc[:, booking_data_preprocessed.columns == 'event_type_enc']
	
	#split dataset intotrain and test for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
	
	#sampling function since dataset is unbalanced
    X_train_oversample, y_train_oversample = sampling_func(X_train,y_train)
	
	#model fit, prediction and evaluation
    y_pred, feature_importance = model_fit(X_train_oversample, X_test, y_train_oversample, y_test)
	
	# summarize the fit of the model
	target_names = ['book', 'not_book']
	print(); print(classification_report(y_test, y_pred,target_names=target_names))
	print(); print(confusion_matrix(y_test, y_pred))
	
    print(feature_importance.loc[feature_importance.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False))
	
	