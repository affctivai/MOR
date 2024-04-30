import os
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.cluster import KMeans

class LeaveOneSubjectOut:
    def __init__(self, split_path, info):
        self.split_path = split_path
        self.info = info
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
            self.split_info_constructor(info)
            print('created in splt_path =', self.split_path)
        else: print('already in splt_path =', self.split_path)
    def split_info_constructor(self, info: pd.DataFrame) -> None:
        subjects = list(set(info['subject_id']))
        for test_subject in subjects:
            train_subjects = subjects.copy()
            train_subjects.remove(test_subject)
            train_info = []
            for train_subject in train_subjects:
                train_info.append(info[info['subject_id'] == train_subject])
            train_info = pd.concat(train_info)
            test_info = info[info['subject_id'] == test_subject]
            train_info.to_csv(os.path.join(self.split_path, f'sub{test_subject}_train.csv'), index=False)
            test_info.to_csv( os.path.join(self.split_path, f'sub{test_subject}_test.csv'), index=False)
        print('split_path =', self.split_path)

    def split(self, dataset, subject):
        train_info = pd.read_csv(os.path.join(self.split_path, f'sub{subject}_train.csv'))
        test_info  = pd.read_csv(os.path.join(self.split_path, f'sub{subject}_test.csv'))
        train_dataset = deepcopy(dataset);  train_dataset.info = train_info
        test_dataset = deepcopy(dataset);   test_dataset.info = test_info
        return train_dataset, test_dataset

class KMeansCrossTrialPerSubject:
    def __init__(self, n_clusters, test_size, seed, split_path, info):
        self.n_clusters = n_clusters
        self.test_size = test_size
        self.seed = seed
        self.split_path = split_path
        self.info = info
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
        
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
            self.split_info_constructor()
            print('Created in split_path =', self.split_path)
        else: print('Already exists in split_path =', self.split_path)

    def split_info_constructor(self):
        subjects = self.info['subject_id'].unique()
        for subject in subjects:
            subject_info = self.info[self.info['subject_id'] == subject]
            subject_info['cluster'] = self.kmeans.fit_predict(subject_info[['valence', 'arousal']])
            train_info, test_info = [], []
            for cluster in range(self.n_clusters):
                cluster_data = subject_info[subject_info['cluster'] == cluster]
                unique_trials = cluster_data['trial_id'].unique()
                if len(unique_trials) > 1: 
                    train_trials, test_trials = model_selection.train_test_split(unique_trials, test_size=self.test_size, random_state=self.seed)    
                    train_data = cluster_data[cluster_data['trial_id'].isin(train_trials)]
                    test_data = cluster_data[cluster_data['trial_id'].isin(test_trials)]
                else:  # 하나의 trial만 있는 경우, 전체를 훈련 데이터로
                    train_data = cluster_data; test_data = pd.DataFrame() 
                train_info.append(train_data); test_info.append(test_data)
            train_info = pd.concat(train_info, ignore_index=True)
            test_info = pd.concat(test_info, ignore_index=True)
            train_info.to_csv(os.path.join(self.split_path, f'sub{subject}_train_C{self.n_clusters}_seed{self.seed}.csv'), index=False)
            test_info.to_csv(os.path.join(self.split_path, f'sub{subject}_test_C{self.n_clusters}_seed{self.seed}.csv'), index=False)

    def split(self, dataset, subject):
        train_info = pd.read_csv(os.path.join(self.split_path, f'sub{subject}_train_C{self.n_clusters}_seed{self.seed}.csv'))
        test_info = pd.read_csv(os.path.join(self.split_path,  f'sub{subject}_test_C{self.n_clusters}_seed{self.seed}.csv'))
        train_dataset = deepcopy(dataset); train_dataset.info = train_info
        test_dataset = deepcopy(dataset);  test_dataset.info = test_info
        return train_dataset, test_dataset