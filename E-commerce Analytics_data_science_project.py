# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:51:46 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:47:45 2022

@author: Admin
Capstone Project
E-commerce Analytics
Building an unspervised learning model and performing RFM analysis to segment customers into groups
"""
 #import libraries
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import calendar
from datetime import timedelta
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

# Read the data


path= 'D:/Capstone project/Ecomdata.xlsx'

data = pd.read_excel(path)


print(data)
 
data.shape

data.columns

data.head()

data.tail()
# correcting columns with spell errors and spaces

data = data.rename(columns=({'CustomerID':'Customer_ID','Item Code':'Item_Code','InvoieNo':'Invoice_No','Date of purchase':'Date_of_purchase','price per Unit':'price_per_Unit','Shipping Location':'Shipping_Location','Reason of return':'Reason_of_return','Sold as set':'Sold_as_set'}))

data.columns


'''OBSERVATIONS
1. CustomerID has null values and is float instead of integer
2. Cancelled_status has values as 'True' for orders than have been cancelled
3. Reason_of_return has only 3 entries with a reason
4. Sold_as_set column is completely empty
'''

data.Customer_ID.isnull().sum() # 133790 (25%) rows do not have a customerID
# We could impute the blank customerID using the unique invoice numbers but this would be incorrect considering the fact that a single customer can have multiple invoices and by imputing these blank values with a unique CustomerID we will assume that these were from individual customers along with giving a heavy biasness to the data by these single invoiced customers. Unfortunately, we will have to bear a data loss in this case so we exclude the rows with a blank Customer ID
# NOTE: We cannot use the assumption as segmentation is to be done using customer ID

data_clean = data.copy()
data_clean = data[data['Customer_ID'].isnull() != True]

data_clean.shape

data_clean= data_clean.reset_index(drop=True)

data_clean.columns

data_clean.head()

data_clean['Customer_ID'] = data_clean['Customer_ID'].astype(int)

data_clean.dtypes

np.round(data_clean['Cancelled_status'].value_counts() / len(data_clean) * 100,1)

# 8183 (2%) are cancelled

'''
1. Since our main objective here is to segment our customers based on their purhcase patterns using the RFM approach, we can exclude the cancellations data and study this separately if required to get a view on customer cancellation and revenue leakage.
2. This leaves us with 74% of the total data
3. We can remove other columns that are blank or will not help us analyse.
4. Remove 'Cancelled_status','Return_reason','Sold_as_set'
'''
data_clean = data_clean[ data_clean['Cancelled_status']!= True]

data_clean.shape

data_clean.columns

print(data_clean)
data_clean.drop(columns=['Cancelled_status','Reason_of_return','Sold_as_set'],inplace=True)

print(data_clean)

data_clean.columns

data_clean.head()
# update index to run loops based on index
data_clean = data_clean.reset_index(drop=True)
data_clean.head()
data_clean.duplicated(keep='first').sum()

print(data_clean)
data_clean.loc[data_clean.duplicated(keep='first')]

data_clean.shape
data_clean.drop_duplicates(keep='first',inplace=True)

# FINDING PATTERNS

for i in data_clean:
    
    print(i,' : ',data_clean[i].nunique())
    
    
data_clean.shape
data_clean.columns
# We have 4324 unique customers who bought 3637 different items across 18305 invoices(unique orders) which were shipped to 20 locations

# TOP and BOTTOM REVENUE GENERATORS by location

data_clean.groupby(['Shipping_Location']).Price.sum().sort_values().plot(kind = 'barh',figsize=(20,10),color='Green')
plt.title('Revenue by Shipping Location',size=25,color='Green')
plt.xlabel('Revenue (In 100Mn)',size= 22)
plt.ylabel('Shipping_Location',size= 22)
plt.legend(loc='best')


np.round(data_clean['Price'][data_clean['Shipping_Location']=='Location 36'].sum() / data_clean['Price'].sum() *100,1)


# Location 36 is our top revenue generator contributing to 93% of our total revenue for the last 13 months.

plt.figure(figsize=(20,10))
plt.subplot(311)

data_clean[data_clean['Shipping_Location'] != 'Location 36'].groupby(['Shipping_Location']).Price.sum().sort_values()[-5:].plot(kind = 'barh',color='Green')
plt.title('Top 5 Revenue generators after Location 36',size=25,color = 'Green')
plt.xlabel('Revenue (In 10Mn)')
plt.ylabel('Shipping_Location')
plt.legend(loc='best')
plt.subplot(313)
data_clean.groupby(['Shipping_Location']).Price.sum().sort_values()[:5].plot(kind = 'barh',color='Red')
plt.title('Bottom 5 Revenue generators',size=25,color='Red')
plt.xlabel('Revenue')
plt.ylabel('Shipping_Location')
plt.legend(loc='best')

# Locations 14, 26, 15 are other major contributors to the business generating a revenue of more than 10 million each

data_clean.columns

data_clean.head()
# REVENUE BY PERIOD
# creating hour, day and month cols to analyse trends of purchase
data_clean['Purchase_year'] = data_clean['Date_of_purchase'].dt.year

data_clean.columns
data_clean['Purchase_month'] = data_clean['Date_of_purchase'].dt.month
data_clean['Purchase_day'] = data_clean['Date_of_purchase'].dt.day

data_clean[['Date_of_purchase','Purchase_year','Purchase_month','Purchase_day']]


# extract hours and dayname
Purchase_hour = []
#for i in range(len(data_clean['Time'])):
for i in list(data_clean.index):
    a = data_clean['Time'][i].hour
    Purchase_hour.append(a)

data_clean['Purchase_hour'] = Purchase_hour


dayname = []
#for i in range(len(data_clean)):
for i in list(data_clean.index):
    a= calendar.day_name[calendar.weekday(data_clean['Purchase_year'][i],data_clean['Purchase_month'][i],data_clean['Purchase_day'][i])]
    dayname.append(a)

data_clean['Purchase_dayname'] = dayname


print(data_clean.Purchase_dayname)

print(data_clean)

data_clean.columns

plt.figure(figsize=(20,10))
plt.subplot(311)
data_clean.groupby(['Purchase_hour']).Price.sum().plot(kind='line')
plt.title('Revenue by hour')
plt.xlabel('Hour of Day')
plt.ylabel('Revenue (In 100Mn)')
plt.xticks(data_clean['Purchase_hour'].unique())
plt.legend(loc='best')
plt.subplot(313)
data_clean.groupby(['Purchase_month']).Price.sum().plot(kind='bar')
plt.title('Revenue by month')
plt.xlabel('Month of year')
plt.ylabel('Revenue (In 100Mn)')
plt.xticks(data_clean['Purchase_month'].unique())
plt.legend(loc='best')

plt.figure(figsize=(20,10))
plt.subplot(511)
data_clean.groupby(['Purchase_dayname']).Price.sum().plot(kind='bar')
plt.title('Revenue by day',fontsize=15)
plt.xlabel('Day of week')
plt.ylabel('Revenue (In 100Mn)')
plt.legend(loc='best')
plt.subplot(515)
data_clean.groupby(['Date_of_purchase']).Price.sum().plot(kind='line')
plt.title('Daily Revenue Trend',fontsize=15)
plt.xlabel('Invoice_date')
plt.ylabel('Revenue (In Mn)')
plt.xticks([])
plt.legend(loc='best')


'''
OBSERVATIONS
1. Major ordering hours are between 8-17 where 10-15 are biggest peak hours.
2. A drop in orders and revenue is seen towards the end of the month (post 24th of every month)
3. A spike in sales is seen since September maintaining high orders and revenue until December probably owing to a festive season
'''

# Seasonal analysis

data_clean['Type'] = data_clean['Date_of_purchase'].dt.month.isin([9,10,11,12])
data_clean['Type'][data_clean['Type']==True] = 'Festive'
data_clean['Type'][data_clean['Type']==False] = 'Non-Festive'

data_clean['Type'].value_counts()

plt.figure(figsize=(20,10))
plt.subplot(131)
data_clean.groupby(['Type']).Price.sum().plot(kind = 'pie',autopct = '%1.0f%%')
plt.title('Revenue: Festive vs Non-Festive',size = 15,color='brown')
plt.xlabel(None)
plt.ylabel(None)
plt.subplot(132)
data_clean.groupby(['Type']).Quantity.sum().plot(kind = 'pie',autopct = '%1.0f%%')
plt.title('Products sold: Festive vs Non-Festive',size = 15,color='Blue')
plt.xlabel(None)
plt.ylabel(None)
plt.subplot(133)
data_clean.groupby(['Type']).Invoice_No.nunique().plot(kind = 'pie',autopct = '%1.0f%%')
plt.title('Orders placed: Festive vs Non-Festive',size = 15,color='Green')
plt.xlabel(None)
plt.ylabel(None)

data_clean.columns

# Nearly 50% of our business is covered during the festive months. we should look at opportunities to boost our sales during the non-festive seasons by promotion offers / retention startergies to increase revenue

# Top and bottom 5 products
plt.figure(figsize=(20,10))
plt.subplot(311)
data_clean.groupby(['Item_Code']).Price.sum().sort_values()[-5:].plot(kind = 'barh',color='Green')
plt.title('Top 5 Revenue generating items',size=15,color='Green')
plt.xlabel('Revenue (In 10Mn)')
plt.ylabel('Shipping_Location')
plt.legend(loc='best')
plt.subplot(313)
data_clean.groupby(['Item_Code']).Price.sum().sort_values()[:5].plot(kind = 'barh',color='Red')
plt.title('Bottom 5 Revenue generating items',size=15,color='Red')
plt.xlabel('Revenue')
plt.ylabel('Shipping_Location')
plt.legend(loc='best')

# RFM calculation

data_clean['Date_of_purchase'] = pd.to_datetime(data_clean['Date_of_purchase'])

data_clean.shape

data_clean.info()

data_clean['total_sum'] = data_clean['Quantity'] * data_clean['price_per_Unit']

print(data_clean['total_sum'])

max_date= data_clean['Date_of_purchase'].max() + timedelta(hours=0,minutes=0,seconds=0)

print(max_date)
data_process = data_clean.groupby(['Customer_ID']).agg({
        'Date_of_purchase': lambda dt: (max_date- dt.max()).days,
        'Invoice_No': 'count',
        'total_sum': 'sum'})
data_clean.columns

data_process.columns

data_process.shape

data_clean.shape
# Rename the columns
data_process.info()

data_process.rename(columns={'Date_of_purchase' : 'Recency','Invoice_No':'Frequency','total_sum':'Monetary'},inplace=True)

data_process.columns

r_labels= range(4,0,-1)
f_labels= range(1,5)
m_labels= range(1,5)

r_groups= pd.qcut(data_process['Recency'],q=4,labels=(r_labels))
f_groups= pd.qcut(data_process['Frequency'],q=4,labels=(f_labels))
m_groups= pd.qcut(data_process['Monetary'],q=4,labels=(m_labels))
data_process= data_process.assign(R=r_groups.values,F=f_groups.values,M=m_groups.values)

print(data_process)

# rfm score

data_process['RFM_score']= data_process[['R','F','M']].sum(axis=1)

print(data_process)

print(data_process.shape)

# rfm level function
#Define rfm_level function
def rfm_level(df):
     if df['RFM_score'] >=9:
         return 'Can not loose them'
     elif((df['RFM_score'] >=8) and (df['RFM_score'] <9)):
         return'Champions' 
     elif((df['RFM_score'] >=7) and (df['RFM_score'] <8)):
         return'Loyal'
     elif((df['RFM_score'] >=6) and (df['RFM_score'] <7)):
         return'Potential'
     elif((df['RFM_score'] >=5) and (df['RFM_score'] <6)):
         return'Promising'
     elif((df['RFM_score'] >=4) and (df['RFM_score'] <5)):
         return'Needs Attention'
     else:
         return'Require Activation'
data_process['RFM_Level'] = data_process.apply(rfm_level, axis=1)


print(data_process)


data_process.to_csv('D:/Imarticus Dsp/Python/Capstone/RFM_data.csv')

data_process.shape

temporary= data_process
print(data_clean)

data_clean.columns

data_clean.shape

data_clean.dtypes 


from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
le = LabelEncoder()
 
# Encode labels in column 'ItemCode'.
data_clean['Item_Code'] = data_clean['Item_Code'].astype(str)
le.fit(data_clean['Item_Code'])
data_clean['Item_Code']= le.transform(data_clean['Item_Code'])

print(data_clean['Item_Code'])
shp_label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'ShippingLocation'.
le.fit(data_clean['Shipping_Location'])
data_clean['Shipping_Location']= le.transform(data_clean['Shipping_Location'])

print(data_clean.Shipping_Location)

print(data_clean)

data_clean.dtypes
# One Condition
# X = data_clean[['Customer_ID','Item_Code','Invoice_No', 'Quantity', 'price_per_Unit', 'Price', 'Shipping_Location'
#        ,'Purchase_hour', 'Purchase_month', 'Purchase_day']]


# Second Condition
X = data_clean[['Customer_ID','Item_Code','Invoice_No', 'Quantity', 'price_per_Unit', 'Shipping_Location'
        ,'Purchase_hour', 'Purchase_month', 'Purchase_day']]








distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(2, 11)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
 
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_


for key, val in mapping1.items():
    print(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

# Final K Means Model
kmeanModel = KMeans(n_clusters=4).fit(X)
kmeanModel.fit(X)
clusters = kmeanModel.labels_
data_clean['cluster_group'] = clusters

unique_cust_id = sorted(list(set(list(data_clean['Customer_ID'].values))))

customer_cluster = []

for each_cust in unique_cust_id:
    temp_df = data_clean.loc[data_clean['Customer_ID']==each_cust, ['cluster_group']]
    cluster_lsit = list(temp_df['cluster_group'].values)
    mode_of_cluster = max(set(cluster_lsit), key = cluster_lsit.count)
    print(mode_of_cluster)
    print(each_cust)
    customer_cluster.append(mode_of_cluster)

# temporary['cluster_group'] = customer_cluster
data_process['cluster_group'] = customer_cluster


len(clusters)

data_clean.shape
data_clean.to_csv('D:/Imarticus Dsp/Python/Capstone/pred_cutomer_clusters.csv')
data_process.to_excel('D:/Imarticus Dsp/Python/Capstone/result.xlsx')

print(data_clean)



