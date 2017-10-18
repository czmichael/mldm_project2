import pandas as pd
import numpy as np

DATA_DIR = '../data'
CSV_FILE = DATA_DIR + '/KaggleV2-May-2016.csv'

# read data using read_csv function
appt_df = pd.read_csv(CSV_FILE, 
                        dtype={ 'Age': np.float64
                              },
                       parse_dates = ['ScheduledDay', 'AppointmentDay'])
appt_df.head()

# do data cleanup here
# highly recommend that you rename dataset
# e.g., if you used appt_df = pd.read_csv(...) above
# first thing to do here is clean_appt_df = appt_df

clean_appt_df = appt_df.copy()
clean_appt_df.describe()

# rename some variables
clean_appt_df = clean_appt_df.rename(index=str, 
                     columns = {"Hipertension": "Hypertension",
                      "Handcap": "Handicap"})

# remove negative ages
clean_appt_df = clean_appt_df.drop(clean_appt_df[clean_appt_df['Age'] < 0].index)

# take another look
clean_appt_df.describe()


# save as csv file
clean_appt_df.to_csv('../processed_data/clean_appt_df.csv', index=False)

PROCESSED_DATA_DIR = '../processed_data'
clean_appt_df = pd.read_csv(PROCESSED_DATA_DIR + "/clean_appt_df.csv",
                           parse_dates=['AppointmentDay', 'ScheduledDay'])
# create test set with stratified sampling here
# refer to the intro notebook posted in class calendar for example

# check class proportions in complete dataset
clean_appt_df['No-show'].value_counts() / len(clean_appt_df)


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=20000, random_state=1234)

for train_index, test_index in split.split(clean_appt_df, clean_appt_df['No-show']):
    train_set = clean_appt_df.iloc[train_index]
    test_set = clean_appt_df.iloc[test_index]
    
# check class proportions on train and test sets to make sure 
# properly stratified

print("Train set:")
print(train_set['No-show'].value_counts() / len(train_set))

print("Test set:")
print(test_set['No-show'].value_counts() / len(test_set))


# save train and test sets as csvs
train_set.to_csv(PROCESSED_DATA_DIR + '/train_set.csv', index=False)
test_set.to_csv(PROCESSED_DATA_DIR + '/test_set.csv', index=False)

# copy data frame to only use train set
clean_appt_df = train_set.copy()
clean_appt_df.shape


# is there a difference in the no show rate based on gender
clean_appt_df.groupby('Gender')['No-show'].value_counts(normalize=True).plot.bar()

# is there a difference in the age distribution
# between classes
clean_appt_df[['Age','No-show']].boxplot(by='No-show')

#======================   Data PreProcessing ============================

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

# github.com/pandas-dev/sklearn-pandas
# install with pip install sklearn-pandas
from sklearn_pandas import DataFrameMapper

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
    ('df_mapper', df_mapper)
])


clean_df = pd.read_csv(PROCESSED_DATA_DIR + "/train_set.csv", parse_dates=['ScheduledDay','AppointmentDay'],
                      dtype={'Age': np.float64})
clean_df_labels = clean_df['No-show'].copy()
clean_df = clean_df.drop('No-show', axis=1)
print(clean_df.head())


full_pipeline.fit(clean_df)
appt_mat = full_pipeline.transform(clean_df)

print(appt_mat[:15,:].toarray())
