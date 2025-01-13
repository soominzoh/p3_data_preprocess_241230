import os
import shutil
import pandas as pd
import numpy as np
from scipy.io import savemat
from datetime import datetime
import re

# 오늘 날짜를 포함한 경로 생성
today_date = datetime.today().strftime('%y%m%d')
Dir_path = r'C:\Users\user\Desktop\천안임상\b\P3\11' #변경해서 사용. 아래 인바디 저장된 path도 변경해야 함
Save_path = r'C:\Users\user\Desktop\천안임상\b\P3\11\tmp'
os.makedirs(Save_path, exist_ok=True)

ra1_files = [f for f in os.listdir(Dir_path) if f.endswith('.ra1')]
raw_files = [f for f in os.listdir(Dir_path) if f.endswith('.raw')]

#ref_data = pd.read_excel('Y:/사내임상실험/Reference_data.xlsx', dtype=str)
#zip함수 활용 ra1_file, raw_file동시에 반복문 돌게 처리
file_pairs = {}
for ra1_file, raw_file in zip(ra1_files, raw_files):
    ra1_data_id = ra1_file[16:27]
    raw_data_id = raw_file[16:27]
    
    if ra1_data_id == raw_data_id:
        if ra1_data_id not in file_pairs:
            file_pairs[ra1_data_id] = []
        file_pairs[ra1_data_id].append((ra1_file, raw_file))

for data_id, pairs in file_pairs.items():
    folder_path = os.path.join(Save_path, data_id)
    os.makedirs(folder_path, exist_ok=True)
    
    raw_folder = os.path.join(folder_path, 'raw')
    csv_folder = os.path.join(folder_path, 'csv')
    txt_folder = os.path.join(folder_path, 'txt')
    mat_folder = os.path.join(folder_path, 'mat')
    inbody_folder = os.path.join(folder_path, 'inbody')
    
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)
    os.makedirs(mat_folder, exist_ok=True)
    os.makedirs(inbody_folder, exist_ok=True)

    #data_id와 file_name의 전화번호 일치하는 파일만 저장
    for file_name in os.listdir(Dir_path):
        full_file_name = os.path.join(Dir_path, file_name)
        if os.path.isfile(full_file_name) and file_name[16:27] == data_id:
            shutil.copy(full_file_name, raw_folder)
    
    for i, (ra1_file, raw_file) in enumerate(pairs):
        ra1_file_path = os.path.join(Dir_path, ra1_file)
        raw_file_path = os.path.join(Dir_path, raw_file)

        with open(ra1_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        sbp = ra1_file[37:40]
        dbp = ra1_file[40:43]
        hr = ra1_file[43:46]
        pwv = ra1_file[-7:-4]

        oscillo_pwv_cal = []
        bandpass_oscillo = []
        skip_next_line = False

        for line in lines[4:]:
            line = line.strip()
            if not line:
                continue
            if line.startswith('-'):
                skip_next_line = True
                continue
            if skip_next_line:
                skip_next_line = False
                continue
            parsed_line = line.split(',')
            if len(parsed_line) == 1:
                oscillo_pwv_cal.append(float(parsed_line[0]))
            elif len(parsed_line) == 3:
                bandpass_oscillo.append(float(parsed_line[2]))

        ra1_meta_data = pd.DataFrame({
            'SBP': [sbp],
            'DBP': [dbp],
            'HR': [hr],
            'PWV': [pwv]
        })

        ra1_df = pd.DataFrame(oscillo_pwv_cal, columns=['oscillo_PWV_cal'])
        bandpass_oscillo_df = pd.DataFrame(bandpass_oscillo, columns=['bandpass_oscillo'])
        ra1_combined_df = pd.concat([ra1_meta_data, ra1_df, bandpass_oscillo_df], axis=1)

        with open(raw_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = [line.strip().split(',') for line in lines[2:]]

        ks_band_pass = []
        ks_raw = []
        raw_oscillo = []

        for row in data:
            if len(row) >= 2:
                ks_band_pass.append(int(row[0], 16))
                ks_raw.append(int(row[1], 16))
                if len(row) == 3 and row[2]:
                    raw_oscillo.append(int(row[2], 16))
                else:
                    raw_oscillo.append(np.nan)


        ks_band_pass = np.array(ks_band_pass)
        ks_raw = np.array(ks_raw)
        raw_oscillo = np.array(raw_oscillo)

        raw_df = pd.DataFrame({
            'ks_band_pass': ks_band_pass,
            'ks_raw': ks_raw,
            'raw_oscillo': raw_oscillo
        })

        combined_df = pd.concat([ra1_combined_df, raw_df], axis=1)

        #SBP, DBP reference data dataframe에 넣는 코드
        #matching_row = ref_data[ref_data['ID'] == data_id]

        # if not matching_row.empty:
        #     stetho_col = f'Stetho_{i+1}'
        #     if stetho_col in matching_row.columns:
        #         if pd.isna(matching_row[stetho_col].values[0]) != True:
        #             sbp_ref, dbp_ref = matching_row[stetho_col].iloc[0].split('/')
        #             combined_df['SBP_ref'] = np.NaN
        #             combined_df['DBP_ref'] = np.NaN

        #             combined_df.loc[0, 'SBP_ref'] = int(sbp_ref)
        #             combined_df.loc[0, 'DBP_ref'] = int(dbp_ref)
        #         else:
        #             combined_df['SBP_ref'] = np.NaN
        #             combined_df['DBP_ref'] = np.NaN

        combined_df.to_csv(os.path.join(csv_folder, f'{data_id}_{i+1}.csv'), index=False)
        combined_df.to_csv(os.path.join(txt_folder, f'{data_id}_{i+1}.txt'), index=False)

        dfm = combined_df.astype(float)
        data_dict = {col : dfm[col].tolist() for col in dfm.columns}
        
        savemat(os.path.join(mat_folder, f'{data_id}_{i+1}.mat'), data_dict)

# #인바디 분리 함수
# def split_dataframe_by_id(df, base_path):
#     for unique_id in df['1. ID'].unique():
#         match = re.match(r'<(\d+)>', unique_id)
#         if match:
#             folder_name = match.group(1)
#         else:
#             folder_name = unique_id

#         df_id = df[df['1. ID'] == unique_id]

#         id_path = os.path.join(base_path, folder_name, 'inbody')
#         if not os.path.exists(id_path):
#             os.makedirs(id_path)

#         file_path = os.path.join(id_path, f'data_{folder_name}.csv')
#         file_path_txt = os.path.join(id_path, f'data_{folder_name}.txt')
        
#         df_id.to_csv(file_path, index=False)
#         df_id.to_csv(file_path_txt, index=False)
#         print(f'Saved {file_path}')

# # 인바디 분리 실행
# inbody_df = pd.read_csv('C:/Users/user/Desktop/Korotkoff/Inbody/inbody970_data.csv')
# split_dataframe_by_id(inbody_df, Save_path)

# print("All files processed.")

#RTasdfasdfadsfdasfdsaf