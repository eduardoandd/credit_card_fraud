import pickle
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score

with open('../credit_card_fraud.pkl','rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste,df=pickle.load(f)

# ============= APLICAÇÃO DO ALGORITMO =============
nave_credit=GaussianNB()
nave_credit.fit(X_treinamento,y_treinamento)
previsao=nave_credit.predict(X_teste)

accuracy_score(y_teste,previsao) #98%
cm=ConfusionMatrix(nave_credit)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)


