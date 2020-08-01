#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[126]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[127]:


# # Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# fifa = pd.read_csv("fifa.csv")

# In[128]:


fifa = pd.read_csv("fifa.csv")


# In[129]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[130]:


fifa.describe()


# In[131]:


fifa.dtypes


# In[132]:


cons = pd.DataFrame({'colunas':fifa.columns,
                'tipo': fifa.dtypes,
                'missing':fifa.isna().sum(),
                'size':fifa.shape[0],
                'unicos':fifa.nunique()})
cons['percentual'] = round(cons['missing']/cons['size'],5)
cons


# In[133]:


cons.percentual.plot.hist()


# In[134]:


print('numero de colunas sem dados faltante',cons[cons.percentual==0].shape[0])
print('numero de colunas com dados faltante',cons[cons.percentual>0].shape[0])


# In[135]:


cons[cons.percentual==0]['tipo'].value_counts()


# In[180]:


fifa=fifa.dropna()


# In[185]:


pca = PCA().fit(fifa)

evr = pca.explained_variance_ratio_
evr[0].round(3)


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[186]:


def q1():
    pca = PCA().fit(fifa)

    evr = pca.explained_variance_ratio_
    return float(round(evr[0],3))


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[143]:


def q2():
    pca_095 = PCA(n_components=0.95)
    X_reduced = pca_095.fit_transform(fifa)

    return X_reduced.shape[1] # Segundo elemento da tupla é o número de componentes encontrados.
    


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[190]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]
myPCA = PCA(n_components=2).fit(fifa)
com = myPCA.components_.dot(x)
tuple(com.round(3))


# In[145]:


def q3():
    myPCA = PCA(n_components=2).fit(fifa)
    com = myPCA.components_.dot(x)
    return tuple(com.round(3))


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[195]:


from sklearn.linear_model import LinearRegression

reg= LinearRegression()

Y_train = fifa['Overall']
X_train = fifa.drop(['Overall'], axis=1)
reg.fit(X_train, Y_train)

colunas_treinamento = X_train.columns

from sklearn.feature_selection import RFE

rfe = RFE(reg, n_features_to_select=5)
rfe.fit(X_train, Y_train)

fifa_selected = pd.DataFrame({'coluna':X_train.columns,
            'escolheu_a_feature': rfe.get_support(),
            'coeficientes': pd.Series(reg.coef_)})

fifa_selected = fifa_selected[fifa_selected['escolheu_a_feature']==True]
list([fifa_selected['coluna']])


# In[124]:


def q4():
    from sklearn.linear_model import LinearRegression

    reg= LinearRegression()

    Y_train = fifa['Overall']
    X_train = fifa.drop(['Overall'], axis=1)
    reg.fit(X_train, Y_train)

    colunas_treinamento = X_train.columns

    from sklearn.feature_selection import RFE

    rfe = RFE(reg, n_features_to_select=5)
    rfe.fit(X_train, Y_train)

    fifa_selected = pd.DataFrame({'coluna':X_train.columns,
                'escolheu a feature': rfe.get_support(),
                'coeficientes': pd.Series(reg.coef_)})

    fifa_selected = fifa_selected[fifa_selected['escolheu a feature']==True]
    return list(fifa_selected['coluna'])

