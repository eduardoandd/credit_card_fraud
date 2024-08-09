import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

# ========== PRÉ PROCESSAMENTO ==========
df=pd.read_csv('fraud_data.csv')
df.isnull().sum()
df['merchant'].duplicated().value_counts()
df.drop(['city','state','trans_num'],axis=1,inplace=True)
df.info()
df['dob']=pd.to_datetime(df['dob'], format='%d-%m-%Y')
df['dob']= (2024 - df['dob'].dt.year)
df=df.rename(columns={'dob':'age'})
df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M').dt.date
# df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'])

# ========== VISUALIZAÇÃO ==========
plt.hist(x=df['age'])
sns.countplot(x=df['merchant'].duplicated(),palette='dark')

# ========== DIVISÃO ENTRE PREVISORES E CLASSE ==========
X=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values

# ============= TRATAMENTO DE ATRIBUTOS CATEGORICOS =============
#LabelEnconder
label_encoder = LabelEncoder()
indices=[]

for i in range(X.shape[1]):
    # if X[:,i].dtype == 'object':
    #     X[:,i]=label_encoder.fit_transform(X[:,1])

    if df.dtypes[i] == 'object':
        indices.append(i)

one_hot_encoder=ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),indices)],remainder='passthrough')
X=one_hot_encoder.fit_transform(X).toarray()

# ============= ESCALONAMENTO DE VALORES =============
scaler=StandardScaler()
X=scaler.fit_transform(X)

# ============= ESCALONAMENTO DE VALORES =============
X_treinamento,X_teste,y_treinamento,y_teste=train_test_split(X,y,test_size=0.3,random_state=0)

# ============= SALVANDO VARIÁVREIS EM DISCO =============
# with open('credit_card_fraud.pkl', mode='wb') as f:
#     pickle.dump([X_treinamento,y_treinamento,X_teste,y_teste,df],f)
