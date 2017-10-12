import pandas as pd
import Constant as const
import os
import matplotlib.pyplot as plt


class Part_I_Exploere(object):

    def load_train_data(self, data_path = const.PROCESSED_DATA_PATH):
        csv_path = os.path.join(data_path, "train.csv")
        na_values = ['N/A']
        return pd.read_csv(csv_path, na_values=na_values, dtype={'Age': int, 'PatientId': str, 'AppointmentID': str})
    

    #def plot_no_show_against_others(self, data):
        #data.plot(x='age', y='patient_id')
        #plt.show()
        #data.describe()
        
        #data.hist(bins=50, figsize=(20,15))
        #plt.show()
        
        
pd.set_option('display.expand_frame_repr', False)
        
exploerer = Part_I_Exploere()
train_data = exploerer.load_train_data()
train_data.drop(train_data.columns[[0]], axis=1, inplace=True)

print(train_data)
#exploerer.plot_no_show_against_others(train_data)


total_count_by_age = [0] * 100 
no_show_count_by_age = [0] * 100 
no_show_ratio_by_age = [0] * 100    

for index, row in train_data.iterrows():
    age = row['age']
    no_show = row['no_show']
    total_count_by_age[age] += 1
    if no_show == 'Yes':
        no_show_count_by_age[age] += 1


for i in range(0, 100):
    if total_count_by_age[i] != 0:
        ratio = float(no_show_count_by_age[i])/total_count_by_age[i]
        no_show_ratio_by_age[i] = ratio
        print("age: {}, total: {}, no_show: {}, ratio: {}".format(i, total_count_by_age[i], no_show_count_by_age[i], ratio))
        




#grouped = train_data.groupby('age')
ages = train_data.groupby('age').groups.keys()
s = train_data.groupby('age').apply(lambda x: x[x['no_show'] == 'Yes']['no_show'].count())


print(ages)
print(s.tolist())
#print(s)
#pd.value_counts().plot(kind="bar")


age_list = pd.DataFrame(
    {
     'no show ratio': no_show_ratio_by_age
    })





#plt.show()
#print(grouped.apply(lambda x: x[x['now_show'] == True].sum()))



age_list.plot(kind='bar', figsize=(14,5))
plt.show()
