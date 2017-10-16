import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer



class Part_I_Exploere(object):
    
    DATA_PATH = "../data/"
    PROCESSED_DATA_PATH = "../processed_data/"

    def load_train_data(self, data_path = PROCESSED_DATA_PATH):
        csv_path = os.path.join(data_path, "train.csv")
        na_values = ['N/A']
        return pd.read_csv(csv_path, na_values=na_values, dtype={'Age': int, 'PatientId': str, 'AppointmentID': str})
    

    def plot_no_show_against_others(self, train_data):
        total_count_by_age = [0] * 100 
        no_show_count_by_age = [0] * 100 
        no_show_ratio_by_age = [0] * 100    
        total_count_by_gender = {'M': 0, 'F': 0}
        no_show_count_by_gender = {'M': 0, 'F': 0}
        no_show_ratio_by_gender = {'M': 0, 'F': 0}
        total_count_by_neighbourhood = {}
        no_show_count_by_neighbourhood = {} 
        no_show_ratio_by_neighbourhood = {}
        total_count_by_hipdertension = [0] * 2 
        no_show_count_by_hipdertension = [0] * 2 
        no_show_ratio_by_hipdertension = [0] * 2 
        total_count_by_diabetes = [0] * 2 
        no_show_count_by_diabetes = [0] * 2 
        no_show_ratio_by_diabetes = [0] * 2 
        total_count_by_sms_received = [0] * 2 
        no_show_count_by_sms_received = [0] * 2 
        no_show_ratio_by_sms_received = [0] * 2
        total_count_by_appointment_hour = {}
        no_show_count_by_appointment_hour = {} 
        no_show_ratio_by_appointment_hour = {}
        total_count_by_appointment_month = {}
        no_show_count_by_appointment_month = {} 
        no_show_ratio_by_appointment_month = {}
        
        for index, row in train_data.iterrows():
            age = row['age']
            gender = row['gender']
            no_show = row['no_show']
            neighbourhood = row['neighbourhood']
            hipertension = row['hipertension']
            diabetes = row['diabetes']
            sms_received = row['sms_received']
            appointment_hour = row['appointment_hour']
            appointment_month = row['appointment_month']
            
            total_count_by_age[age] += 1
            total_count_by_gender[gender] += 1
            
            if neighbourhood in total_count_by_neighbourhood:
                total_count_by_neighbourhood[neighbourhood] += 1
            else:
                total_count_by_neighbourhood[neighbourhood] = 0
                no_show_count_by_neighbourhood[neighbourhood] = 0
                no_show_ratio_by_neighbourhood[neighbourhood] = 0
                
            if appointment_hour in total_count_by_appointment_hour:
                total_count_by_appointment_hour[appointment_hour] += 1
            else:
                total_count_by_appointment_hour[appointment_hour] = 1
                no_show_count_by_appointment_hour[appointment_hour] = 0 
                no_show_ratio_by_appointment_hour[appointment_hour] = 0
            
            if appointment_month in total_count_by_appointment_month:
                total_count_by_appointment_month[appointment_month] += 1
            else:
                total_count_by_appointment_month[appointment_month] = 1
                no_show_count_by_appointment_month[appointment_month] = 0 
                no_show_ratio_by_appointment_month[appointment_month] = 0            
            
            
                
            total_count_by_hipdertension[hipertension] += 1
            total_count_by_diabetes[diabetes] += 1
            total_count_by_sms_received[sms_received] += 1     
              
                
            if no_show == 'Yes':
                no_show_count_by_age[age] += 1
                no_show_count_by_gender[gender] += 1
                no_show_count_by_neighbourhood[neighbourhood] += 1
                no_show_count_by_hipdertension[hipertension] += 1
                no_show_count_by_diabetes[diabetes] += 1
                no_show_count_by_sms_received[sms_received] += 1
                no_show_count_by_appointment_hour[appointment_hour] += 1
                no_show_count_by_appointment_month[appointment_month] += 1
                
        for i in range(0, 100):
            if total_count_by_age[i] != 0:
                no_show_ratio_by_age[i] = float(no_show_count_by_age[i]) / total_count_by_age[i]
                #print("age: {}, total: {}, no_show: {}, ratio: {}".format(i, total_count_by_age[i], no_show_count_by_age[i], ratio))
        
        
        for i in range(0, 2):
            no_show_ratio_by_hipdertension[i] = float(no_show_count_by_hipdertension[i]) / total_count_by_hipdertension[i]
            no_show_ratio_by_diabetes[i] = float(no_show_count_by_diabetes[i]) / total_count_by_diabetes[i]
            no_show_ratio_by_sms_received[i] = float(no_show_count_by_sms_received[i]) / total_count_by_sms_received[i]
            #print('hipertension: {}, total: {}, no show: {}'.format(i, total_count_by_hipdertension[i], no_show_count_by_hipdertension[i]))
        
        
        no_show_ratio_by_male = float(no_show_count_by_gender['M']) / total_count_by_gender['M']
        no_show_ratio_by_female = float(no_show_count_by_gender['F']) / total_count_by_gender['F']
        no_show_ratio_by_gender['M'] = no_show_ratio_by_male
        no_show_ratio_by_gender['F'] = no_show_ratio_by_female
        #print(total_count_by_gender['F'])
        
        for key in total_count_by_neighbourhood:
            no_show_ratio_by_neighbourhood[key] = float(no_show_count_by_neighbourhood[key])/total_count_by_neighbourhood[key]
            #print("neighbourhood:{}, total: {}, now show: {}".format(key, total_count_by_neighbourhood[key], no_show_count_by_neighbourhood[key]))
        
        for key in total_count_by_appointment_hour:
            no_show_ratio_by_appointment_hour[key] = float(no_show_count_by_appointment_hour[key])/total_count_by_appointment_hour[key]
            #print("hour:{}, total: {}, now show: {}".format(key, total_count_by_appointment_hour[key], no_show_count_by_appointment_hour[key]))

        for key in total_count_by_appointment_month:
            no_show_ratio_by_appointment_month[key] = float(no_show_count_by_appointment_month[key])/total_count_by_appointment_month[key]
        
        fig, axes = plt.subplots(nrows=4, ncols=2)

        
        age_list = pd.DataFrame({
            'age': no_show_ratio_by_age
        })
        age_list.plot(ax=axes[0,0])
        
        gender_list = pd.DataFrame({
            'gender': no_show_ratio_by_gender
        })
        gender_list.plot(ax=axes[1,0], kind='bar')
        
        neighbourhood_list = pd.DataFrame({
            'neighbourhood': no_show_ratio_by_neighbourhood
        })
        neighbourhood_list.plot(ax=axes[2,0])
        
        hipdertension_list = pd.DataFrame({
            'hipdertension': no_show_ratio_by_hipdertension
        })
        hipdertension_list.plot(ax=axes[0,1], kind='bar')
        
        diabetes_list = pd.DataFrame({
            'diabetes': no_show_ratio_by_diabetes
        })
        diabetes_list.plot(ax=axes[1,1], kind='bar')
        
        diabetes_list = pd.DataFrame({
            'sms_received': no_show_ratio_by_sms_received
        })
        diabetes_list.plot(ax=axes[2,1], kind='bar')
        
        appointment_hour_list = pd.DataFrame({
            'appointment_hour': no_show_ratio_by_appointment_hour
        })
        appointment_hour_list.plot(ax=axes[3,0])
        
        appointment_month_list = pd.DataFrame({
            'appointment_month': no_show_ratio_by_appointment_month
        })
        appointment_month_list.plot(ax=axes[3,1], kind='bar')        
        
        plt.suptitle('no show ratio')
        plt.show()
                
                
                
                
                
    def preprocess(self, df):
        le_neib = LabelEncoder()
        df["neighbourhood_code"] = le_neib.fit_transform(df["neighbourhood"])
        print(df.dtypes)
       
        lb_gender = LabelBinarizer()
        lb_results = lb_gender.fit_transform(df["gender"])
        
        df['is_male'] = lb_results        
        print(df)
                
                
# ============== Main =========================
                
pd.set_option('display.expand_frame_repr', False)
        
exploerer = Part_I_Exploere()
train_data = exploerer.load_train_data()
train_data.drop(train_data.columns[[0]], axis=1, inplace=True)

print(train_data)
exploerer.plot_no_show_against_others(train_data)
exploerer.preprocess(train_data)


