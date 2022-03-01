import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

class KFold:
    def __init__(self, n_splits=5, 
                csv_path='/opt/ml/input/data/train/label_train.csv', 
                img_path='/opt/ml/input/data/train/images',
                random_state=1004):
        self.csv_path = csv_path
        self.img_path = img_path
        self.folds =[]
        self._generate_kfold(n_splits, random_state)

    def __len__(self):
        return len(self.folds)
    
    def __getitem__(self, idx):
        return self.folds[idx]

    def _generate_kfold(self, n_splits, random_state):
        df = pd.read_csv(self.csv_path)

        # df = self._data_preprocessing(df)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_idx, test_idx in skf.split(df, df["age"] + 3 * df["gender"]):
            # augment한 60대 데이터를 train에 기존 데이터를 val에 넣는 작업
            trans_train = train_idx
            trans_test = test_idx
            for t in train_idx:
                tmp = df.loc[t]['path'].split('_')
                if tmp[3] == '60':
                    if len(tmp) != 5:
                        trans_train.remove(t)
                        trans_test.append(t)
            for v in test_idx:
                tmp = df.loc[v]['path'].split('_')
                if tmp[3] == 60:
                    if len(tmp) == 5:
                        trans_test.remove(t)
                        trans_train.append(t)
            train_idx = trans_train
            test_idx = trans_test

            df_strat_train = self._generate_path_and_mask_and_label_field(df.loc[train_idx])
            df_strat_test = self._generate_path_and_mask_and_label_field(df.loc[test_idx])
            
            self.folds.append([df_strat_train.reset_index(), df_strat_test.reset_index()])

    def get_preprocessed_df(self, aug_csv_path):
        df = pd.read_csv(aug_csv_path)

        age_label = [0, 29, 59, 120]
        df['age'] = pd.cut(df['age'], age_label, labels=False)

        gender_label = {'male':0, 'female':1}
        df['gender'] = df['gender'].map(gender_label)

        df = self._generate_path_and_mask_and_label_field(df)
        return df


    def _data_preprocessing(self, df):
        # age & gender categorize
        df = self._correct_age_gender_label(df)

        age_label = [0, 29, 59, 120]
        df['age'] = pd.cut(df['age'], age_label, labels=False)

        gender_label = {'male':0, 'female':1}
        df['gender'] = df['gender'].map(gender_label)
        return df

    def _correct_age_gender_label(self, df):
        gender_status_invalid = ["000225", "000664", "000767", "001720", "001498-1", "001509", 
                                "003113", "003223", "004281", "004432", "005223", "006359", 
                                "006360","006361", "006362", "006363", "006364", "006424"]
        gender_change = {'male':'female', 'female':'male'}
        
        df.loc[df['id'].isin(gender_status_invalid), 'gender'] = df.loc[df['id'].isin(gender_status_invalid), 'gender'].map(gender_change)

        # 59 to 60
        index = df[df['id'] == "004348"].index
        df.loc[index, 'age'] = 60
        return df
    
    def _correct_mask_label(self, df):
        mask_status_invalid = ["000020", "004418", "005227"]
        mask_change = {0:0, 1:2, 2:1}

        df.loc[df['id'].isin(mask_status_invalid), 'mask'] = df.loc[df['id'].isin(mask_status_invalid), 'mask'].map(mask_change)

        return df

    def _generate_path_and_mask_and_label_field(self, df):
        mask_mapping = {'incorrect_mask':1, 'mask1':0, 'mask2':0,
                        'mask3':0, 'mask4':0, 'mask5':0, 'normal':2}
        df['mask'] = [['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal'] for _ in range(len(df))]
        df = df.explode('mask')
        
        df['path'] = df.apply(lambda row: self._search_image(row['path'], row['mask']), axis=1)
        
        df['mask'] = df['mask'].map(mask_mapping)

        # df = self._correct_mask_label(df)
        df['label'] = df['mask'] * 6 + df['gender'] * 3 + df['age']
        return df

    def _search_image(self, path, mask):
        
        for f in os.listdir(os.path.join(self.img_path, path)):
            if f.startswith(mask):
                return os.path.join(self.img_path, path, f)

        raise FileNotFoundError