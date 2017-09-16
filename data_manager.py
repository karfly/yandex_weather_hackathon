import os
from os.path import join as pj

import random

import numpy as np
np.random.seed = 0  # for reproducibility

import pandas as pd

import json
from tqdm import tqdm

from sklearn.neighbors import KDTree


class DataManager(object):
    def __init__(self,
                 raw_data_path='./data/raw/data',
                 city_name='spb',
                 nrows=None,  # read first nrow of train data
                 ):
        self.raw_data_path = raw_data_path
        self.city_name = city_name
        self.nrows = nrows

        self._setup_paths()
        self.X_train, self.y_train, self.X_val, self.y_val = self._preprocess_train()
        self.X_test = self._preprocess_test()

    def _setup_paths(self):
        self.train_path = pj(self.raw_data_path, 'train_{}.tsv'.format(self.city_name))
        self.train_netatmo_path = pj(self.raw_data_path, 'train_{}_netatmo.tsv'.format(self.city_name))
        self.train_col_dtypes_path = pj(self.raw_data_path, 'train_col_dtypes.json')

        self.test_path = pj(self.raw_data_path, 'test_{}_features.tsv'.format(self.city_name))
        self.test_netatmo_path = pj(self.raw_data_path, 'test_{}_netatmo.tsv'.format(self.city_name))
        self.test_col_dtypes_path = pj(self.raw_data_path, 'test_col_dtypes.json')

        self.hackathon_tosubmit_path = pj(self.raw_data_path, 'hackathon_tosubmit.tsv')

    def _preprocess_train(self):
        # train df
        print('Loading train df...')
        self.df_train = pd.read_csv(self.train_path, sep='\t',dtype=json.load(open(self.train_col_dtypes_path)),
                                    nrows=self.nrows)

        # netatmo df
        print('Loading train netatmo df...')
        self.df_train_netatmo = pd.read_csv(self.train_netatmo_path, na_values='None', sep='\t', dtype={'hour_hash': 'uint64'},
                                            nrows=self.nrows)

        print('Preprocessing train netatmo df...')
        self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree = DataManager._preprocess_netatmo(self.df_train_netatmo)

        # extracting train features
        print('Extracting features...')
        train_by_group = self.df_train.groupby(["city_code","sq_x","sq_y","hour_hash"])
        X, y, block_ids = [], [], []

        for block_id in tqdm(train_by_group.groups):
            group = train_by_group.get_group(block_id)
            X.append(self._extract_features_from_group(group))
            y.append(group.iloc[0]['rain'])
            block_ids.append(block_id + (group.iloc[0]["hours_since"],))  # for validation

        del self.df_train
        del self.df_train_netatmo, self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree

        X = pd.DataFrame(X).fillna(-999.)
        y = np.array(y)
        block_ids = pd.DataFrame(block_ids, columns=["city_code","sq_x","sq_y","hour_hash","hours_since"])

        # train/val split
        in_train = block_ids['hours_since'] <= np.percentile(block_ids['hours_since'], 85) #leave last 15% for validation
        
        print(np.unique(block_ids['hours_since']))
        X_train, y_train = X[in_train], y[in_train]
        X_val, y_val = X[~in_train],y[~in_train]
        print("Training samples: %i; Validation samples: %i" % (len(X_train),len(X_val)))

        return X_train, y_train, X_val, y_val

    def _preprocess_test(self):
        # test df
        print('Loading test df...')
        self.df_test = pd.read_csv(self.test_path, sep='\t',dtype=json.load(open(self.test_col_dtypes_path)),
                                   nrows=self.nrows)

        print('Loading test netatmo df')
        self.netatmo_df_test = pd.read_csv(self.test_netatmo_path, na_values="None", sep='\t',dtype={'hour_hash': "uint64"})
        self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree = DataManager._preprocess_netatmo(self.netatmo_df_test)

        # extracting test features
        print('Extracting features...')
        test_by_group = self.df_test.groupby(["city_code","sq_x","sq_y","hour_hash"])

        X_test, test_block_ids = [],[]
        for block_id in tqdm(test_by_group.groups):
            group = test_by_group.get_group(block_id)
            X_test.append(self._extract_features_from_group(group))
            test_block_ids.append(block_id)

        del self.df_test 
        del self.netatmo_df_test, self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree
            
        X_test = pd.DataFrame(X_test)
        test_block_ids = pd.DataFrame(test_block_ids,columns=["city_code","sq_x","sq_y","hour_hash"])

        return X_test


    @staticmethod
    def _preprocess_netatmo(netatmo_df):  
        df_by_hour = netatmo_df.groupby('hour_hash')
        netatmo_hour_hash_to_kdtree = {}
        for hour, stations_group in df_by_hour:
            netatmo_hour_hash_to_kdtree[hour] = KDTree(stations_group[["netatmo_latitude","netatmo_longitude"]].values,
                                                       metric='minkowski', p=2)
        
        # convert groupby to dict to get faster queries
        netatmo_hour_hash_to_data = {group: stations_group for group, stations_group in df_by_hour}
        
        return netatmo_hour_hash_to_data, netatmo_hour_hash_to_kdtree


    def _extract_features_from_group(self, group):
        """
        group is df_train.groupby(["city_code","sq_x","sq_y","hour_hash"]) group
        """

        features = {}

        # square features
        square = {col: group[col].iloc[0] for col in group.columns}
        
        features['_square_lat'] = square['sq_lat']
        features['_square_lon'] = square['sq_lon']
        features['_day_hour'] = square['day_hour']

        # signal strength
        features['_signal_mean'] = group['SignalStrength'].mean()
        features['_signal_var'] = group['SignalStrength'].var()

        # features for each user
        group_by_user = group.groupby('u_hashed')
        # group_by_user.apply(lambda group: group['ulat'].var()+group['ulon'].var())
        
        features['_num_users'] = len(group_by_user)
        features['_mean_entries_per_user'] = group_by_user.apply(len).mean()
        features['_mean_user_signal_var'] = group_by_user.apply(
            lambda user_entries: user_entries['SignalStrength'].var()).mean()
        
        # netatmo features
        if square['hour_hash'] in self.netatmo_hour_hash_to_data:
            local_stations, neighbors = self.netatmo_hour_hash_to_data[square['hour_hash']], self.netatmo_hour_hash_to_kdtree[square['hour_hash']]
            [distances], [neighbor_ids] = neighbors.query([(square['sq_lat'], square['sq_lon'])],k=10)

            neighbor_stations = local_stations.iloc[neighbor_ids]

            features['_netatmo_distance_to_closest_station'] = np.min(distances)
            features['_netatmo_mean_distance_to_station'] = np.mean(distances)

            for colname in ['netatmo_pressure_mbar','netatmo_temperature_c','netatmo_sum_rain_24h',
                            'netatmo_humidity_percent',"netatmo_wind_speed_kmh","netatmo_wind_gust_speed_kmh"]:
                col = neighbor_stations[colname].dropna()
                if len(col)!=0:
                    features['_netatmo_' + colname + "_mean"], features['_netatmo_' + colname + "_mean"] = col.mean(), col.var()
                else:
                    features['_netatmo_' + colname + "_mean"], features['_netatmo_' + colname + "_mean"] = np.nan, np.nan

        return features