#Impoting module
import os
from PIL import Image  
import pandas as pd
import json

path = 'EVA_Assignment_22'
files = os.listdir(path)
i = 1

#Rename
for file in files:
    if (i<10):
        os.rename(file,"img_00" + str(i) +'.jpg')
    else:
        os.rename(file,"img_0" + str(i) +'.jpg')
    i = i+1    

#Resize    
files = os.listdir(path)

for file in files:
    img = Image.open(file)
    img = img.resize((400,400))
    img.save(file)

#Retriving height and width values from JSON
with open('Annotation.json') as json_file:
    data = json.load(json_file)

img_data = data['_via_img_metadata']
width_list = []
height_list = []

for key,value in img_data.items():
    if('img' in key):
        for i,j in value.items():
            if(isinstance(value[i],list)):
                for i in value[i]:
                    for a,b in i.items():
                        for c,d in b.items():
                            if('width' in c):
                                width_list.append(d)
                            elif('height' in c):
                                height_list.append(d)


#Converting list into dataframe
hw = pd.DataFrame()
divvariable = 400
height_list = [x / divvariable for x in height_list]
width_list = [x / divvariable for x in width_list]
hw['height'] = height_list
hw['width'] = width_list


#Normalizing DATASET
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaled = scaler.fit_transform(hw)

#Applying K-MEANS
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(hw)

#Finding cluster centers
print(kmeans.cluster_centers_)

#Finding INVERSE_TRANSFORM of cluster centroids 
#unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
