import pandas as pd
import numpy as np


PROCESSED_DATA_DIR = '../processed_data'
#======================   Data PreProcessing ============================

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

# github.com/pandas-dev/sklearn-pandas
# install with pip install sklearn-pandas
from sklearn_pandas import DataFrameMapper



class YesNoTransform(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        no_show = np.ndarray(shape=X.shape, dtype=int)
        for index, rowdata in X.iterrows():
            if any(rowdata == 'Yes'):
                no_show[index] = 1
            if any(rowdata == 'No'):
                no_show[index] = -1
        return no_show


yes_no_mapper = DataFrameMapper([
    (['No-show'], YesNoTransform())
], input_df=True)
    
    
no_show_pipeline = Pipeline([
    ('yes_no_mapper', yes_no_mapper)
])    


class WeekdayTransform(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X['AppointmentDay'].dt.weekday.values

weekday_mapper = DataFrameMapper([
    (['AppointmentDay'], WeekdayTransform())
], input_df=True)
    

weekday_pipeline = Pipeline([
    ('weekday_adder', weekday_mapper),
    ('weekday_encoder', OneHotEncoder(n_values=7))
])


class DaysAheadTransform(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        daysahead = (X['AppointmentDay'] - X['ScheduledDay'])\
            .dt.days\
            .values\
            .astype('float64')
        return daysahead
    
daysahead_mapper = DataFrameMapper([
    (['AppointmentDay', 'ScheduledDay'], DaysAheadTransform())
], input_df=True)

daysahead_pipeline = Pipeline([
    ('mapper', daysahead_mapper),
    ('scaler', StandardScaler())
])

date_pipeline = FeatureUnion(transformer_list=[
    ('weekday_pipeline', weekday_pipeline),
    ('daysahead_pipeline', daysahead_pipeline)
])

numeric_attributes = ['Scholarship',
                      'Hypertension',
                      'Diabetes',
                      'Alcoholism',
                      'SMS_received'
                     ]

num_mapper = DataFrameMapper(list(zip(numeric_attributes, [None for x in numeric_attributes])))

df_mapper = DataFrameMapper([
    (['Age'], StandardScaler()),
    ('Gender', LabelBinarizer()),
    ('Neighbourhood', LabelBinarizer()),
    (['Handicap'], OneHotEncoder())
])


full_pipeline = FeatureUnion(transformer_list=[
    ('date_pipeline', date_pipeline),
    ('num_mapper', num_mapper),
    ('df_mapper', df_mapper),
    ('no_show_pipeline', no_show_pipeline)
])


clean_df = pd.read_csv(PROCESSED_DATA_DIR + "/train_set.csv", parse_dates=['ScheduledDay','AppointmentDay'],
                      dtype={'Age': np.float64})
print(clean_df['No-show'].value_counts())
#clean_df_labels = clean_df['No-show'].copy()
#clean_df = clean_df.drop('No-show', axis=1)

#print(clean_df.head())


full_pipeline.fit(clean_df)
appt_mat = full_pipeline.transform(clean_df)
print(type(appt_mat))

appt_mat_df = pd.DataFrame(appt_mat.toarray())

appt_mat_df.to_csv(PROCESSED_DATA_DIR + '/appt_mat.csv', index=False)
print(appt_mat[:5,:].toarray())
print(appt_mat.shape)
#print(clean_df_labels[:5])






