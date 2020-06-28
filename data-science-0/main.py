#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[95]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[6]:


def q1():
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    return black_friday.loc[(black_friday['Gender']=='F') & (black_friday['Age']=='26-35')].shape[0]
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    return len(black_friday['User_ID'].unique())
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[163]:


len(black_friday.dtypes.unique())


# In[7]:


def q4():
    return int(len(black_friday.dtypes.unique()))


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[215]:


(black_friday.shape[0] - black_friday.dropna().shape[0])/ black_friday.shape[0]


# In[71]:


def q5():
    return (black_friday.shape[0] - black_friday.dropna().shape[0])/ black_friday.shape[0]
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[74]:


def q6():
    return int((black_friday.isna().sum()[black_friday.isna().sum()>0]).max())
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    return int(black_friday['Product_Category_3'].dropna().mode())
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


purchase_norm =pd.DataFrame()
purchase_norm = (black_friday['Purchase'] - black_friday['Purchase'].min())/(black_friday['Purchase'].max()-black_friday['Purchase'].min())


# In[11]:


def q8():
    mean = (purchase_norm).mean()
    return float(mean)
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[249]:


purchase_pad =pd.DataFrame()
purchase_pad = (black_friday['Purchase'] - black_friday['Purchase'].mean())/black_friday['Purchase'].std()

((purchase_pad<1) & (purchase_pad>-1)).sum() 


# In[12]:


def q9():
    return int(((purchase_pad<1) & (purchase_pad>-1)).sum() )
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[149]:


#total de valores na
black_friday[['Product_Category_2', 'Product_Category_3']].isna().sum()


# In[157]:


#quando as duas variaveis sao na
df_na =  black_friday[['Product_Category_2', 'Product_Category_3']].isna()
df_na.loc[(df_na['Product_Category_2']==True) & (df_na['Product_Category_3']==True)].sum()


# Ou seja, sempre que Product_Category_2 é na então Product_Category_3 é na

# In[13]:


def q10():
    return True
    pass

