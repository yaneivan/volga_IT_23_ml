import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# temp = pd.read_csv(os.path.join('mnt', 'source', 'temperature.csv'), delimiter=';', encoding='utf-8')


temp = pd.read_csv(os.path.join('mnt', 'source', 'temperature.csv'), delimiter=';', encoding='utf-8')
with open(os.path.join('..', 'model', 'houses.pkl'), 'rb') as f:
    loaded = pickle.load(f)

houses = loaded['houses']
# houses.head()

new_temp = []

current_date = 0
temps=[]
humids=[]
for index, row in temp.iterrows():
    if row['date_start'].split()[0] != current_date:
        new_temp.append([current_date, np.array(temps), np.array(humids)])
        current_date = row['date_start'].split()[0]
        temps = []
        humids = []
    temps.append(float(row['temp'].replace(',', '.')))
    humids.append(row['humidity'])
    

    
# print(new_temp, len(temps))
new_temp = pd.DataFrame(new_temp, columns=['date', 'temp', 'humidity'])
temp = new_temp.drop(axis = 0, index=0)
# temp

model = nn.Sequential(
    nn.Linear(32, 128),
    nn.ReLU(),
    nn.Linear(128, 64), 
    nn.ReLU(),
    nn.Linear(64, 1)
)
model.load_state_dict(torch.load(os.path.join('..', 'model', 'model.pt')))


result = []

print(houses.shape)
for index_h, house_row in tqdm(houses.iterrows(), total=houses.shape[0]):
    summa = 0
    
    for index_t, temp_row in temp.iterrows():
        index
        a = temp_row['temp'].tolist()[:8]
        b = temp_row['humidity'].tolist()[:8]
        c = house_row.iloc[1:].tolist()
        while (len(a)<8):
            a += [0]
        while (len(b) < 8):
            b += [0]
        
        x = ([  a + b + c]) 
        x = loaded['scaler'].transform(x)
        x = np.insert(x, 0, 0, axis=1)
        x = torch.FloatTensor(x)
        pred = model(x)
        pred = float(pred.cpu().detach()[0][0])
        summa += pred
        
    result.append([house_row['address_uuid'], summa])
#         pred = loaded['temp_scaler'].inverse_transform(pred.cpu().detach())
#         print(len(row['temp'].tolist()), len(row['humidity'].tolist()), len(house_row.iloc[1:].tolist()))
#         print(len(x[0]))
        
#             tmp = (((houses.loc[houses['address_uuid'] == row['address_uuid']]).iloc[:, 1:]))
            
#             df.append([   row['value'], row['is_unreliable']] + 
#                       (((temp.loc[temp['date'] == row['date']])['temp']).tolist()[0].tolist()) + 
#                       ((temp.loc[temp['date'] == row['date']])['humidity']).tolist()[0].tolist() +  
#                       (tmp.loc[tmp.index[0]]).tolist()
                      


result = pd.DataFrame(result, columns=['address_uuid', 'volume'])


result.to_csv(os.path.join('mnt', 'result', 'volume.csv'))