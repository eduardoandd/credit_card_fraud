import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from yellowbrick.classifier import ConfusionMatrix
import pandas as pd
import matplotlib.pyplot as plt

with open('../naive_bayes/credit_card_fraud.pkl','rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste,df=pickle.load(f)

arvore= DecisionTreeClassifier(criterion='entropy')
arvore.fit(X_treinamento,y_treinamento)
arvore.feature_importances_
previsao=arvore.predict(X_teste) 
accuracy_score(y_teste,previsao) #97%

cm=ConfusionMatrix(arvore)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)

previsores= df.columns[:-1].tolist()
figura,eixos=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
tree.plot_tree(arvore,filled=True)
