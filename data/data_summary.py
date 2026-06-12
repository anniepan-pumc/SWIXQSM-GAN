import pandas as pd
import numpy as np
import os

data_path = '/Volumes/MOVESPEED/QSM_multicohort_data/ukb_validate2'
record_path = '/Users/panxingqi/Documents/Code/CVMed/data/ukb_data_summary_validate2.csv'
SWI_path = ''
QSM_path = ''

data_record = pd.read_csv(record_path, dtype={'SWI': str, 'QSM': str, 'InBoth': int})
data_record = data_record.loc[data_record['InBoth'] == 1]
df = pd.DataFrame(columns=['SWI', 'QSM', 'InBoth'])
for i,row in data_record.iterrows():
    swi_path = os.path.join(data_path, row['SWI'], 'SWI.nii.gz')
    qsm_path = os.path.join(data_path, row['QSM'], 'QSM.nii.gz')
    if os.path.exists(swi_path) and os.path.exists(qsm_path):
        print(row['SWI']," SWI and QSM exist", swi_path, qsm_path)
        df = pd.concat([df, pd.DataFrame({'SWI': row['SWI'], 'QSM': row['QSM'], 'InBoth': 1}, index=[0])], ignore_index=True)
    else:
        print(row['SWI']," SWI or QSM not exist", swi_path, qsm_path)
        # df = pd.concat([df, pd.DataFrame({'SWI': row['SWI'], 'QSM': row['QSM'], 'InBoth': 0}, index=[0])], ignore_index=True)
df.to_csv('ukb_data_summary_new.csv', index=False)