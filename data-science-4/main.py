#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1246]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)


# In[1247]:


# # Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[1248]:


countries = pd.read_csv("countries.csv")


# In[1249]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[1250]:


countries.describe()


# In[1251]:


cons = pd.DataFrame({'colunas': countries.columns,
                    'tipos':countries.dtypes,
                    'faltantes': countries.isna().sum(),
                    'tamanho': countries.shape[0],
                    'unicos': countries.nunique()})
cons['percentual'] = round(cons['faltantes']/cons['tamanho'],5)
cons


# In[1252]:


try:
    for coluna in cons['colunas']:
        print(coluna)
        if  coluna=='Population' or coluna=='Area' or coluna=='GDP':
            continue 
        if coluna=='Region' or coluna=='Country':
            countries[coluna] = [str.strip(x)  for x in countries[coluna]]
            continue

        countries[coluna] = [(str(x).replace(',', '.')) for x in countries[coluna]]
        countries[coluna] = countries[coluna].astype('float64')

except (AttributeError):
    print("Ajustes ok")


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[1254]:


def q1():
    regions = countries['Region'].unique()
    return sorted(regions)


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[1255]:


countries_c = countries.copy()

discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
discretizer.fit(countries_c[["Pop_density"]])
score_bins = discretizer.transform(countries[["Pop_density"]])
sum(score_bins[:,0]==9)


# In[1256]:


def get_interval(bin_idx, bin_edges):
  return f"{np.round(bin_edges[bin_idx], 2):.2f} ⊢ {np.round(bin_edges[bin_idx+1], 2):.2f}"

bin_edges_quantile = discretizer.bin_edges_[0]

print(f"Bins quantile")
print(f"interval: #elements\n")
for i in range(len(discretizer.bin_edges_[0])-1):
    print(f"{get_interval(i, bin_edges_quantile)}: {sum(score_bins[:, 0] == i)}")

score_intervals = pd.Series(score_bins.flatten().astype(np.int)).apply(get_interval, args=(bin_edges_quantile,))


# In[1257]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    discretizer.fit(countries_c[["Pop_density"]])
    score_bins = discretizer.transform(countries_c[["Pop_density"]])
    return sum(score_bins[:,0]==9)


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[1259]:


one_hot_encoder_sparse = OneHotEncoder(sparse=True) # sparse=True é o default.

countries_c['Climate']=countries_c['Climate'].fillna(0)

region_encoded_sparse = one_hot_encoder_sparse.fit_transform(countries_c[["Region"]])
print(region_encoded_sparse.toarray().shape)


climate_encoded_sparse = one_hot_encoder_sparse.fit_transform(countries_c[["Climate"]])
print(climate_encoded_sparse.toarray().shape)


# In[1260]:


def q3():
   outliers = region_encoded_sparse.toarray().shape[1] + climate_encoded_sparse.toarray().shape[1]
   return outliers


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[1262]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ('std_scaler',StandardScaler())
])

pipeline_transformation = num_pipeline.fit(countries_c.iloc[:,2:])
# num_pipeline.get_params()
countries.head(5)


# In[1263]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[1264]:


transform_test = num_pipeline.transform([test_country[2:]])


# In[1265]:


def q4():
    return float(transform_test[:, countries_c.columns.get_loc('Arable') - 2].round(3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[1267]:


countries_outlier=countries['Net_migration']


# In[1269]:


sns.boxplot(countries_outlier, orient="vertical")


# In[1270]:


sns.distplot(countries_outlier.fillna(0))


# In[1271]:


qan1 = countries_outlier.quantile(0.25)
qan3 = countries_outlier.quantile(0.75)
iqr = qan3 - qan1

non_outlier_interval_iqr = [qan1 - 1.5 * iqr, qan3 + 1.5 * iqr]

print(f"Faixa considerada \"normal\": {non_outlier_interval_iqr}")


# In[1272]:


outliers_bellow = countries_outlier[(countries_outlier < non_outlier_interval_iqr[0])] 
outliers_above = countries_outlier[(countries_outlier > non_outlier_interval_iqr[1])] 


# In[1273]:


def q5():
    return (len(outliers_bellow), len(outliers_above), False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[1275]:


from sklearn.datasets import fetch_20newsgroups 
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
len(newsgroup.data)


# In[1276]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)
type(newsgroups_counts)


# In[1277]:


phone_idx = count_vectorizer.vocabulary_.get("phone")
sum(newsgroups_counts[:,phone_idx].toarray())


# In[1278]:


def q6():
    return int(sum(newsgroups_counts[:,phone_idx].toarray()))


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[1280]:


def q7():
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_vectorizer.fit(newsgroup.data)

    newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroup.data)

    sum(newsgroups_tfidf_vectorized[:,phone_idx].toarray())

    return float(sum(newsgroups_tfidf_vectorized[:,phone_idx].toarray()).round(3))


# In[ ]:




