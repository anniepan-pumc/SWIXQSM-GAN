import pandas as pd
import os

data_path = '/Volumes/MOVESPEED/QSM_multicohort_data/ukb_validate2'
SWI_path = 'SWI.nii.gz'
QSM_path = 'QSM/QSM.nii.gz'

df = pd.DataFrame(columns=['SWI', 'QSM', 'InBoth'])
for sub in os.listdir(data_path):
    swi_path = os.path.join(data_path, sub, SWI_path)
    qsm_path = os.path.join(data_path, sub, QSM_path)
    if os.path.exists(swi_path) and os.path.exists(qsm_path):
        df = pd.concat([df, pd.DataFrame({'SWI': sub, 'QSM': sub, 'InBoth': 1}, index=[0])], ignore_index=True)
    else:
        df = pd.concat([df, pd.DataFrame({'SWI': sub, 'QSM': sub, 'InBoth': 0}, index=[0])], ignore_index=True)
df.to_csv('ukb_data_summary_validate2.csv', index=False)