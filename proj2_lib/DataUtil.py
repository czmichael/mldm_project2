import pandas as pd
import numpy as np
import os 
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


class DataUtil(object):

    DATA_PATH = "../data/"
    PROCESSED_DATA_PATH = "../processed_data/"
    
    def load_data(self, data_path = DATA_PATH):
        csv_path = os.path.join(data_path, "KaggleV2-May-2016.csv")
        na_values = ['N/A']
        return pd.read_csv(csv_path, na_values=na_values, dtype={'Age': int, 'PatientId': str, 'AppointmentID': str})
    
    def clean_data(self, data):
        scheduled_month_list = []
        scheduled_hour_list = []
        appointment_month_list = []
        appointment_hour_list = []
        
        
        for index, row in data.iterrows():
            gender = row['Gender']
            if gender != 'F' and gender != 'M':
                print('error')
            
            scheduled_day = row['ScheduledDay'] 
            scheduled_day_obj = time.strptime(scheduled_day, "%Y-%m-%dT%H:%M:%SZ")
            scheduled_month = time.strftime("%m", scheduled_day_obj)
            scheduled_month_list.append(scheduled_month)
            scheduled_hour = time.strftime("%H", scheduled_day_obj)
            scheduled_hour_list.append(scheduled_hour)
            data.set_value(index, 'ScheduledDay', time.strftime("%Y-%m-%d %H:%M:%S", scheduled_day_obj))
            
            appointment_day = row['AppointmentDay'] 
            appointment_day_obj = time.strptime(scheduled_day, "%Y-%m-%dT%H:%M:%SZ")
            appointment_month = time.strftime("%m", appointment_day_obj)
            appointment_month_list.append(appointment_month)
            appointment_hour = time.strftime("%H", appointment_day_obj)
            appointment_hour_list.append(appointment_hour)
            data.set_value(index, 'AppointmentDay', time.strftime("%Y-%m-%d %H:%M:%S", appointment_day_obj))
            
            
                            
            age = row['Age']
            if age < 0 or age > 200:
                data.set_value(index, 'Age', -1)
                

        data.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day',  
                        'appointment_day', 'age', 'neighbourhood', 'scholarship', 
                        'hipertension', 'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']  
        
        data['scheduled_month'] = scheduled_month_list
        data['scheduled_hour'] = scheduled_hour_list
        data['appointment_month'] = appointment_month_list
        data['appointment_hour'] = appointment_hour_list
        '''
        data.columns = ['patient_id', 'appointment_id', 'gender', 'scheduled_day',
                        'appointment_day', 'appointment_month', 'appointment_hour', 'age', 'neighbourhood', 'scholarship', 
                        'hipertension', 'diabetes', 'alcoholism', 'handicap', 'sms_received', 'no_show']        
        '''
        print(data)
        
        return data        
     
    def create_test_data(self, data):
        print(data['no_show'].value_counts() / len(data))   
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9047, random_state=0)
        
        for train_index, test_index in sss.split(data, data["no_show"]):
            train_set = data.loc[train_index]
            test_set = data.loc[test_index]
        
        print(train_set)
        train_set.to_csv(self.PROCESSED_DATA_PATH + 'train.csv')
        test_set.to_csv(self.PROCESSED_DATA_PATH + 'test.csv')
        
        print(train_set["no_show"].value_counts() / len(train_set))      
        print(test_set["no_show"].value_counts() / len(test_set)) 

    
    
# ============== Main =========================

pd.set_option('display.expand_frame_repr', False)
            
data_util = DataUtil()

data = data_util.load_data()
#print(data.info())

#with pd.option_context('display.max_rows', None):



clean_data = data_util.clean_data(data)
print(clean_data)
data_util.create_test_data(clean_data)