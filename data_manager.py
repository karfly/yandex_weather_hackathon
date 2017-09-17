import os
from os.path import join as pj

import random

import numpy as np
np.random.seed(0)  # for reproducibility

import pandas as pd

import json
from tqdm import tqdm

from sklearn.neighbors import KDTree

import collections
from functools import partial

from joblib import Parallel, delayed
import multiprocessing

def applyParallel(dfGrouped, func):
    # retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(dfGrouped.get_group(block_id)) for block_id in dfGrouped.groups)
    # return retLst
    return pd.DataFrame(retLst)


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
        self._preprocess_train()
        self._preprocess_test()

        self._postprocess_nans()

    def _extract_features_from_group(self, group):
        """
        group is df_train.groupby(["city_code","sq_x","sq_y","hour_hash"]) group
        """

        def add_statistics_of_features(features, name, array, prefix='', without=[]):
            try:
                array = array.dropna()
            except:
                pass

            statnames_and_fuctions = [
                ('mean', np.mean),
                ('var', np.var),
                ('min', np.min),
                ('max', np.max),
                ('num', len),
                ('5p', partial(np.percentile, q=5)),
                # ('10p', partial(np.percentile, q=10)),
                ('15p', partial(np.percentile, q=15)),
                ('85p', partial(np.percentile, q=85)),
                # ('90p', partial(np.percentile, q=90)),
                ('95p', partial(np.percentile, q=95)),
            ]

            for statname, function in statnames_and_fuctions:
                colname = '{}{}_{}'.format(prefix, name, statname)
                if len(array) == 0 or statname in without:
                    features[colname] = np.nan
                else:
                    features[colname] = function(array)

            return features

        def add_statistics_of_cat_features(features, name, array, prefix='', without=[]):
            try:
                array = array.dropna()
            except:
                pass


            statnames_and_fuctions = [
                ('popular', lambda array: collections.Counter(array).most_common()[0][0]),
                ('num', len),
            ]

            for statname, function in statnames_and_fuctions:
                colname = '{}{}_{}_cat'.format(prefix, name, statname)
                if len(array) == 0 or statname in without:
                    features[colname] = np.nan
                else:
                    features[colname] = function(array)

            return features

        features = {}

        # square features
        square = {col: group[col].iloc[0] for col in group.columns}
        
        features['square_lat'] = square['sq_lat']
        features['square_lon'] = square['sq_lon']
        features['time_of_day'] = square['day_hour']

        # sq_x, sq_y
        features['_square_x'] = square['sq_x']
        features['_square_y'] = square['sq_y']

        # cat features
        features = add_statistics_of_cat_features(features, 'cell_hash', group['cell_hash'], prefix='_')
        features = add_statistics_of_cat_features(features, 'radio', group['radio'], prefix='_')
        features = add_statistics_of_cat_features(features, 'LAC', group['LAC'], prefix='_')
        features = add_statistics_of_cat_features(features, 'LocationPrecision', group['LocationPrecision'], prefix='_')
        features = add_statistics_of_cat_features(features, 'OperatorID', group['OperatorID'], prefix='_')
        features = add_statistics_of_cat_features(features, 'device_model_hash', group['device_model_hash'], prefix='_')


        # statistics of features
        features = add_statistics_of_features(features, 'SignalStrength', group['SignalStrength'])
        features = add_statistics_of_features(features, 'LocationAltitude', group['LocationAltitude'], prefix='_')
        features = add_statistics_of_features(features, 'LocationSpeed', group['LocationSpeed'], prefix='_')
        features = add_statistics_of_features(features, 'LocationDirection', group['LocationDirection'], prefix='_')
        features = add_statistics_of_features(features, 'EventTimestampDelta', group['EventTimestampDelta'], prefix='_')
        features = add_statistics_of_features(features, 'range', group['range'], prefix='_')
        features = add_statistics_of_features(features, 'ulat', group['ulat'], prefix='_')
        features = add_statistics_of_features(features, 'ulon', group['ulon'], prefix='_')
        features = add_statistics_of_features(features, 'cell_lat', group['cell_lat'], prefix='_')
        features = add_statistics_of_features(features, 'cell_lon', group['cell_lon'], prefix='_')

        # features for each user
        group_by_user = group.groupby('u_hashed')

        features = add_statistics_of_features(
            features, 'var_lat_lon_by_user',
            group_by_user.apply(lambda group: group['ulat'].var() + group['ulon'].var()),
            prefix='_'
        )
        
        features['num_users'] = len(group_by_user)

        features = add_statistics_of_features(
            features, 'entries_per_user',
            group_by_user.apply(len),
            prefix='_'
        )

        features = add_statistics_of_features(
            features, 'user_signal_var',
            group_by_user.apply(lambda user_entries: user_entries['SignalStrength'].var()),
            prefix='_'
        )
        
        
        # netatmo features
        if square['hour_hash'] in self.netatmo_hour_hash_to_data:
            local_stations, neighbors = self.netatmo_hour_hash_to_data[square['hour_hash']], self.netatmo_hour_hash_to_kdtree[square['hour_hash']]
            [distances], [neighbor_ids] = neighbors.query([(square['sq_lat'], square['sq_lon'])], k=10)

            neighbor_stations = local_stations.iloc[neighbor_ids]

            features['netatmo_distance_to_closest_station'] = np.min(distances)
            features['netatmo_mean_distance_to_station'] = np.mean(distances)

            for colname in ['netatmo_pressure_mbar','netatmo_temperature_c','netatmo_sum_rain_24h',
                            'netatmo_humidity_percent',"netatmo_wind_speed_kmh","netatmo_wind_gust_speed_kmh",
                            'netatmo_timestamp_delta', 'netatmo_latitude', 'netatmo_longitude',
                            'netatmo_sum_rain_1h', 'netatmo_wind_direction_deg', 'netatmo_wind_gust_direction_deg',
                            'point_latitude', 'point_longitude']:
                col = neighbor_stations[colname]
                features = add_statistics_of_features(features, colname, col, prefix='_', without=['num'])

            # cat
            for colname in ['netatmo_uid']:
                col = neighbor_stations[colname].dropna()
                features = add_statistics_of_cat_features(features, colname, col, prefix='_', without=['num'])

        return features

    def _setup_paths(self):
        self.train_path = pj(self.raw_data_path, 'train_{}.tsv'.format(self.city_name))
        self.train_netatmo_path = pj(self.raw_data_path, 'train_{}_netatmo.tsv'.format(self.city_name))
        self.train_col_dtypes_path = pj(self.raw_data_path, 'train_col_dtypes.json')

        self.test_path = pj(self.raw_data_path, 'test_{}_features.tsv'.format(self.city_name))
        self.test_netatmo_path = pj(self.raw_data_path, 'test_{}_netatmo.tsv'.format(self.city_name))
        self.test_col_dtypes_path = pj(self.raw_data_path, 'test_col_dtypes.json')

        self.hackathon_tosubmit_path = pj(self.raw_data_path, 'hackathon_tosubmit.tsv')

    def _postprocess_nans(self):
        print('Postprocessing nans...')
        self.X_train = self.X_train.fillna(-999999.)
        self.X_test = self.X_test.fillna(-999999.)

    def _preprocess_nans(self, df):
        df.loc[df['radio'] == -999, 'radio'] = np.nan
        df.loc[df['LocationSpeed'] == -999, 'LocationSpeed'] = np.nan
        df.loc[df['LAC'] == 65535, 'LAC'] = np.nan
        df.loc[df['LocationPrecision'] == -21912, 'LocationPrecision'] = np.nan
        df.loc[df['range'] == -999, 'range'] = np.nan
        df.loc[df['LocationDirection'] == -999, 'LocationDirection'] = np.nan
        df.loc[df['cell_lon'] == -999, 'cell_lon'] = np.nan
        df.loc[df['cell_lat'] == -999, 'cell_lat'] = np.nan
        df.loc[df['SignalStrength'] == -2147483647, 'SignalStrength'] = np.nan

        return df

    def _preprocess_nans_netatmo(self, df):
        df.loc[df['netatmo_wind_direction_deg'] == -1, 'netatmo_wind_direction_deg'] = np.nan

    def _preprocess_train(self):
        # train df
        print('Loading train df...')
        self.df_train = pd.read_csv(self.train_path, sep='\t',dtype=json.load(open(self.train_col_dtypes_path)),
                                    nrows=self.nrows)
        self.df_train = self._preprocess_nans(self.df_train)      

        # netatmo df
        print('Loading train netatmo df...')
        self.df_train_netatmo = pd.read_csv(self.train_netatmo_path, na_values='None', sep='\t', dtype={'hour_hash': 'uint64'},
                                            nrows=self.nrows)
        self._preprocess_nans_netatmo(self.df_train_netatmo)

        print('Preprocessing train netatmo df...')
        self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree = DataManager._preprocess_netatmo(self.df_train_netatmo)

        # extracting train features
        print('Extracting features...')
        train_by_group = self.df_train.groupby(["city_code","sq_x","sq_y","hour_hash"])
        X, y, block_ids = [], [], []
        # X = applyParallel(train_by_group, self._extract_features_from_group)
        for block_id in tqdm(train_by_group.groups):
            group = train_by_group.get_group(block_id)
            X.append(self._extract_features_from_group(group))
            y.append(group.iloc[0]['rain'])
            block_ids.append(block_id + (group.iloc[0]["hours_since"],))  # for validation

        del self.df_train
        del self.df_train_netatmo, self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree
        
        X = pd.DataFrame(X)
        y = np.array(y)
        block_ids = pd.DataFrame(block_ids, columns=["city_code","sq_x","sq_y","hour_hash","hours_since"])

        self.X_train, self.y_train = X, y
        self.train_block_ids = block_ids

    def _preprocess_test(self):
        # test df
        print('Loading test df...')
        self.df_test = pd.read_csv(self.test_path, sep='\t',dtype=json.load(open(self.test_col_dtypes_path)),
                                   nrows=self.nrows)
        self.df_test = self._preprocess_nans(self.df_test)  

        print('Loading test netatmo df')
        self.netatmo_df_test = pd.read_csv(self.test_netatmo_path, na_values="None", sep='\t',dtype={'hour_hash': "uint64"})
        self._preprocess_nans_netatmo(self.netatmo_df_test)
        self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree = DataManager._preprocess_netatmo(self.netatmo_df_test)

        # extracting test features
        print('Extracting features...')
        test_by_group = self.df_test.groupby(["city_code","sq_x","sq_y","hour_hash"])

        X_test, test_block_ids = [],[]
        # X_test = applyParallel(test_by_group, self._extract_features_from_group)
        for block_id in tqdm(test_by_group.groups):
            group = test_by_group.get_group(block_id)
            X_test.append(self._extract_features_from_group(group))
            test_block_ids.append(block_id)

        del self.df_test 
        del self.netatmo_df_test, self.netatmo_hour_hash_to_data, self.netatmo_hour_hash_to_kdtree
            
        self.X_test = pd.DataFrame(X_test)
        self.test_block_ids = pd.DataFrame(test_block_ids,columns=["city_code","sq_x","sq_y","hour_hash"])


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