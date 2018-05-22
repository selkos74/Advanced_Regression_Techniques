
# coding: utf-8

# In[1]:


import numpy as np #biblioteka za rad sa MATRICAMA
import pandas as pd #biblioteka za rad sa DATAFRAME
import matplotlib.pyplot as plt #biblioteka za vizuelizaciju. PROSTIJA VIZUELIZACIJA
import seaborn as sns #biblioteka za vizuelizaciju. LEPSA VIZUELIZACIJA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
get_ipython().magic(u'matplotlib inline')

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF


import plotly.figure_factory as pfc

import scipy
from scipy import stats

from sklearn import preprocessing
#SHIFT + ENTER za izbacivanje OUTPUT-a


# # 1. Preprocesiranje
# Cilj preprocesiranja jeste da se dataset spremi za "učenje" tako što ćemo ga "očistiti" od podataka koji bi mogli da nam predstavljaju problem.

# ## 1.1. Dataset

# In[2]:


train = pd.read_csv('train.csv') #učitavanje training set-a.
test = pd.read_csv('test.csv') #učitavanje test set-a.
testID = test['Id']

data = pd.concat([train.drop('SalePrice', axis=1), test], keys=['train', 'test']) #concat = spajanje.
#dropujemo "SalePrice" iz training set-a jer je to potrebno PREDVIVETI.
data.drop(['Id'], axis=1, inplace=True) #takodje dropujemo ID jer nam on trenutno nije neophodan.


# In[3]:


data.head(2)


# In[4]:


corrmat = train.drop(["Id", "SalePrice"], axis=1).corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[5]:


tempdf = pd.melt(corrmat.reset_index(), id_vars=['index'], value_vars=corrmat.index)


# In[6]:


tempdf[(tempdf.value > 0.8) & (tempdf.value != 1)].drop_duplicates()


# In[7]:


data = data.drop(['GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageCars'],  axis=1)


# In[8]:


data.shape


# #### Anomalije
# 
# Pre nego što nastavimo sa procesiranjem podataka, potrebno je da izvršimo validaciju, tj.da proverimo da li postoje podaci koji nisu realni, tj.koji nemaju smisla. Npr.atributi koji se odnose na godinu čija je vrednost veća od 2018. Takođe cene, razdaljine, površina ne smeju uzimati negativne vredosnti kao što i meseci moraju biti između
# 1 i 12.

# In[9]:


years = ['YearBuilt', 'YearRemodAdd', 'YrSold'] #Svi atributi koji u sebi imaju podatak o GODINI
metrics = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
          '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
         'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'] #.... o VELIČINAMA


# Provera da li postoji neka obzervacija kod koje je GODINA veća od 2018-e.

# In[10]:


data[years].max() #maksimalne vredosti za svaki od atributa koji u sebi sadrže godinu.


# In[11]:


mask = (data[years] > 2018).any(axis=1)
data[mask] 


# Postoji jedna obzervacija kod koje atribut(feature) "GarageYrBlt 2207.0" iskače iz normale.

# In[12]:


data.loc[mask, 'GarageYrBlt'] = data[mask]['YearBuilt']


# Provera da li postoje NEGATIVNE VREDNOSTI za veličinu parcela,garaže,trema itd.

# In[13]:


mask = (data[metrics] < 0).any(axis=1) #provera da li postoji neka veličina koja iskače iz normale.
data[mask]


# Provera da li za sve obzervacije važi da MESECI uzimaju vrednosti od 1 do 12.

# In[14]:


mask = ((data['MoSold'] > 12) | (data['MoSold'] < 1))
data[mask]


# #### Tipovi podataka
# Kada je reč o tipovima podataka, postoje 4 velike grupe:
# a.Kontinualni numerički atributi(dužine,površine,cene)
# b.Diskretni numerički atributi - ne možemo da ih tretiramo kao numeričke atribute jer podržavaju poređenje i aritmetičke operacije(numerički rezultati,broj soba,godine)
# c.Ordinalne kategoričke atribute - atribute koji daju kvalitativne/opisne rezultate("Dobar","Odličan") i koji podržavaju poređenja ali ne i aritmetičke operacije
# d.Čisto kategoričke podatke
# 
# Odlučila sam da u potpunosti odvojim numeričke(i kontinualne i diskretne) od kategoričkih atributa pri čemu zadržavamo vezu između diskretnih numeričkih podataka i radićemo sa manje atributa.
# 

# In[15]:


#Numerički podaci
num_feats = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
             'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 
             'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
             'HeatingQC', '2ndFlrSF', 'LowQualFinSF',
             'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
             'Fireplaces', 'FireplaceQu',
              'GarageArea', 'GarageQual', 'GarageCond',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal',
             'YrSold']  

#Moramo da KONVERTUJEMO ono što je odgovoreno REČIMA("Odlično","Vrlo Dobro", "Dobro") za atribute u
#"grades" u NUMERIČKU skalu
grades = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
          'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
literal = ['Ex', 'Gd', 'TA', 'Fa', 'Po'] #Ex-Excelent, Gd-Good, Po-Poor
num = [9, 7, 5, 3, 2]

G = dict(zip(literal, num))

data[grades] = data[grades].replace(G)


#Kategorički podaci podrazumevaju sve ono što nisu numerički podaci
cat_feats = data.drop(num_feats, axis=1).columns 


# In[16]:


#data[grades]


# In[17]:


cat_feats


# ## 1.2. Normalnost i spljoštenost

# Većina regresionih modela mnogo efikasnije radi sa varijablama koje imaju normalnu(ili približno normalnu) raspodelu.
# Napomena: U obzir nećemo uzimati diskretne numeričke atribute jer će, bez njihove analize, rezultati biti mnogo čiljiviji i pored toga, mnogi od diskretnih numeričkih atributa uzimaju samo nekoliko različitih vredosti tako da je normalnost veoma retka.

# In[18]:


#log transformacija onoga što se želi predvideti.
price = np.log1p(train['SalePrice'])


# In[19]:


#log transformacija spljoštenih kontinualnih numeričkih vrednosti
skewed_feats = data.loc['train'][metrics].apply(lambda x: x.skew(skipna=True)) #izračunavanje SPLJOŠTENOSTI(skewness)
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

data[skewed_feats] = np.log1p(data[skewed_feats])


# In[20]:


#skewed_feats


# ## 1.3. Nedostajuće vredosti
# 
# U posmatranom dataset-u, postoje dva tipa nedostajućih vredosti. Prvi se odnosi na neke nedostajuće vredosti koje nisu zabaležene(mali broj njih) dok se drugi odnosi na nedostajući atribut za neku od obzervacija(kuća nema podrum ili garažu).

# In[21]:


data.isnull().sum()[data.isnull().sum() > 0] #atributi koji imaju barem jednu nedostajuću vrednost


# ##### MSZoning, Utilities, Exterior1st, Exterior2nd, Electrical, Functional and SaleType

# In[22]:


feats = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'Functional',
         'SaleType']
model = data.loc['train'].groupby('Neighborhood')[feats].apply(lambda x: x.mode().iloc[0]) 
#Grupišemo po "Neighborhood"i uzimamo MODE(Najčešća pojavljivanja) i nad time data.iloc[0] -first row of data frame

for f in feats:
    data[f].fillna(data['Neighborhood'].map(model[f]), inplace=True)
    #postavljamo najučestaliju vrednost umesto nedostajućih vredosnti


# In[23]:


#model


# ##### Lot frontage

# In[24]:


data["LotConfig"].value_counts()


# In[25]:


plt.subplots(figsize=(15,5))
boxdata = data.loc['train'].groupby('LotConfig')['LotFrontage'].median().sort_values(ascending=False)
order = boxdata.index

sns.boxplot(x='LotConfig', y='LotFrontage', order=order, data=data.loc['train'])


# In[26]:


boxdata


# In[27]:


data['LotFrontage'] = data['LotFrontage'].fillna(data.loc['train', 'LotFrontage'].median())
#nedostajuće vredosti popunjavamo sa MEDIJANOM(srednjom vrednošću).


# ##### KitchenQual

# Jako je malo nedostajućih vrednosti kada je reč o atributu "KitchenQual" pa ćemo ga zameniti sa vrednosšću za atribut "OverallQual"

# In[28]:


data['KitchenQual'].fillna(data['OverallQual'], inplace=True)


# ##### Podrumi, garaže, kamini i ostali atributi

# Nedostajuće atribute(NA) ćemo nadomestiti tako što ćemo reći da kuća ne poseduje dotične atribute.

# In[29]:


bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath',
        'BsmtHalfBath', 
        'TotalBsmtSF']
fire = ['Fireplaces', 'FireplaceQu']
garage = ['GarageQual', 'GarageCond', 'GarageType', 'GarageFinish', 
          'GarageArea']
masn = ['MasVnrType', 'MasVnrArea']
others = ['Alley', 'Fence', 'PoolQC', 'MiscFeature']

cats = data.columns[data.dtypes == 'object'] #Atributi ciji je datatype OBJECT
nums = list(set(data.columns) - set(cats))

#Kategorija "None" je takođe obrađena
data['MasVnrType'].replace({'None': np.nan}, inplace=True)

data[cats] = data[cats].fillna('0')
data[nums] = data[nums].fillna(0)


# In[30]:


data.isnull().sum().sum()


# ### Izmena tipova varijabli

# Nakon analize, došla sam do zaključka da neki atributi nisu prezentovani odgovarajućim tipom:
# "MSSubClass"-prikazan je kao INTEGER a predstavlja kategorički podatak tako da ćemo koristiti OBJECT.
# "MoSold"-prikazan je kao INTEGER dok mesec predstavlja kategoričku promenljivu sa 12 mogućnosti tako da ćemo koristiti OBJECT.
# "BsmtFullBath" i "BsmtHalfBath" predstavljeni su kao FLOAT a mi ćemo koristiti INTEGER.
# "years" je predstavljen kao FLOAT a mi ćemo koristiti INTEGER.
# "GarageCars"-prikazan je kao FLOAT a mi ćemo koristiti INTEGER.

# In[31]:


data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False) #int u object
data['MoSold'] = data['MoSold'].astype('object', copy=False) #int u object 
data['BsmtFullBath'] = data['BsmtFullBath'].astype('int64', copy=False) #float u int
data['BsmtHalfBath'] = data['BsmtHalfBath'].astype('int64', copy=False) #float u int
data[years] = data[years].astype('int64', copy=False) #float u int


# ### Grupisanje u posebnu kategoriju

# Neke od kategorija kategoričkih atributa su jako loše reprezentovane te se iz njih ne može izvući nikakav zaključak. U cilju rešavanja ovog problema, sve problematične kategorije ćemo svrstati u jednu posebnu kategoriju.

# In[32]:


categorical_data = pd.concat((data.loc['train'][cat_feats], price), axis=1)


# In[33]:


low = 0.05 * data.loc['train'].shape[0] #barem 5% dataset-a treba da ima ovu vrednost

for feat in cat_feats:        
    #grupisaćemo kategorije na osnovu prosečne prodajne cene(average sale price)-mean()SalePrice
    #za svako pojavljivanje feat iz cat_feats, uzmi mi prosek za SalePrice i poredaj u opadajućem redosledu.
    order = ((categorical_data.groupby(feat).mean()).sort_values(by='SalePrice', 
                                                      ascending=False).index.values.tolist())
    
    #order - array(['Normal', 'Abnorml', 'Partial'], dtype=object)
    
    for i in range(0, len(order)): # i između 1 i 3
        N = (categorical_data[categorical_data[feat] == order[i]]
             .count().max()) # N=137
        j = i # j=2 i="Partial"
        while (N < low) & (N != 0): #Dok god postoji atribut čija je vrednost manja od 5% i različita je od 0
            j += 1 #...povećavaj mi kategoriju za 1

            if (j > len(order) - 1):
                # Ako smo stigli do kraja liste, vrati se na poslednju korisnu kategoriju
                # iz "order" liste
                j = i - 1
                break
            else: 
                N += (categorical_data[categorical_data[feat] == order[j]]
                      .count().max())
        if j < i:
            lim = len(order)
        else:
            lim = j

        for k in range(i, lim):
            categorical_data.replace({feat: {order[k]: order[j]}},
                                 inplace=True)
            data.replace({feat: {order[k]: order[j]}},
                                     inplace=True)            
    uniD = data[feat].unique()
    order = categorical_data[feat].unique()

    for i in uniD:
        if i not in order:
            ind = np.argsort(order - i)[0]
            data.replace({feat: {i: order[ind]}}, inplace=True)


# In[34]:


data.columns


# ## 1.4. Dummy encoding

# Prvo moramo da prebacimo atribute sa 2 kategorije u 0-1 encoding. Automatsko korišćenje funkcije get_dummies bi ih konvertovala u dva odvojena atributa(feat_0 i feat_1)).

# In[35]:


#Izmeštamo kolone sa samo jednom kategorijom
for feat in categorical_data.columns[:-1]: 
    uni = categorical_data.groupby(feat).mean().sort_values(by='SalePrice').index
    if (len(uni) < 2):
            data.drop(feat, axis=1, inplace=True) #Izmeštamo kolone sa samo jednom kategorijom
    elif len(uni) < 3: #kolona sa dve kategorije
        print("{}: {}".format(feat, uni))
        data[feat].replace({uni[0]: 0, uni[1]: 1}, inplace=True)
        data[feat] = data[feat].astype('int8')
    else:
        data[feat] = data[feat].astype('category')


# In[36]:


finaldata = pd.get_dummies(data)


# Napomena: Varijable koje sadrže(prikazuju) da "kuća ne poseduje dotične atribute"(odnosi se na garažu, kamin, podrum), neće treitrati "0" kao normalnu kategoriju. Zato ćemo nulu da enkodiramo i za druge mogućnosti.

# In[37]:


black_list = bsmt + fire + garage + masn + others
for feat in finaldata.columns:
    if ('_0' in feat) and (feat.split("_")[0] in black_list):
        finaldata.drop(feat, axis=1, inplace=True)


# In[38]:


finaldata.shape #(broj obzervacija, broj atributa)


# ## 1.5. Shapiro-Wilk test

# ##### FinalData : Shapiro-Wilk test za Normalnu raspodelu

# In[39]:


for column in finaldata.columns:
    test_result = scipy.stats.shapiro(
        finaldata.loc['train'][column]
    )
    print(column)
    print(test_result)
    if test_result[1] > 0.05:
        print("Usvajamo H0!\n")
    else:
        print("Ne usvajamo H0!\n")


# ## 1.6. Feature Scaling

# U cilju efikasnijeg i boljeg rada algoritma linearne regresije, ono što ćemo sledeće da uradimo to je da ćemo da razdvojimo i normalizujemo podatke. Ono što ćemo koristiti za normalizaciju je SREDNJA VREDNOST(mean) i STANDARDNA DEVIJACIJA(standard deviation) našeg training set-a.

# In[40]:


# Training/testing sets za LINEARNU REGRESIJU bez STANDARDIZCIJE za Ridge sa STANDARDIZACIJOM
X_test = finaldata.loc['test']
X_train = finaldata.loc['train']

Y_train = price


# In[41]:


#m = X_train.mean() #Prosečna vrednost parametra u traingn set-u
#m1 = X_test.mean()
#X_train = (X_train - m) / std
#X_test = (X_test - m) / std
#def IQR(X_test):
#    return np.percentile(X_test, 75) - np.percentile(X_test, 25)
#Interquartile range
#def IQR(X_train):
   # return np.percentile(X_train, 75) - np.percentile(X_train, 25)
#X_trainRidge = (X_train - m) / IQR(X_train)
#X_testRidge = (X_test - m) / IQR(X_train)



# # 2. Linearna Regresija

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)


# In[44]:


#Kreiramo LinearRegression objekat
LR = LinearRegression()

#Treniramo model Linearne Regresije korišćenjem training set-a
LR.fit(x_train, y_train)


# # Top influencers

# In[45]:


maxcoef = np.argsort(-np.abs(LR.coef_))
coef = LR.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))


# In[46]:


predictions = LR.predict(x_test)


# In[47]:


diff = pd.DataFrame(predictions - y_test)


# In[48]:


diff.columns = ["diff"]


# In[49]:


diff["squared"] = diff["diff"].apply(lambda x: x*x)


# In[50]:


diff.head()


# In[51]:


np.sqrt(diff["squared"].sum()/diff.shape[0])


# # RMSE = 0.12540078668228263

# ### VIF

# In[52]:


from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ##### finaldata

# In[53]:


vif = pd.DataFrame()


# In[54]:


finaldata.shape


# In[55]:


vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]


# In[56]:


vif["features"] = X_train.columns


# In[57]:


drop_list = vif[vif["VIF Factor"] == np.Inf].features.values


# In[58]:


drop_list #Drop svih onih atributa kod kojih je vrednost za VIF = inf


# In[59]:


X_train_1st_drop = X_train.drop(drop_list, axis=1)


# In[60]:


vif_2nd = pd.DataFrame()


# In[61]:


vif_2nd["VIF Factor"] = [variance_inflation_factor(X_train_1st_drop.values, i) for i in range(X_train_1st_drop.shape[1])]


# In[62]:


vif_2nd["features"] = X_train_1st_drop.columns


# In[63]:


vif_2nd.sort_values("VIF Factor", ascending=False).round(1)


# In[64]:


vif[vif["VIF Factor"] < 5.00].shape # <5 sa INF


# In[65]:


vif_2nd[vif_2nd["VIF Factor"] < 5.00].shape # <5 bez INF


# # Ponovno kreiranje modela - model koji se koristi!

# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(X_train_1st_drop, Y_train, test_size=0.25, random_state=42)


# In[68]:


#LinearRegression object
LR = LinearRegression()

#Treniramo model Linearne Regresije korišćenjem training set-a
LR.fit(x_train2, y_train2)


# ### Top Influencers

# In[69]:


maxcoef = np.argsort(-np.abs(LR.coef_))
coef = LR.coef_[maxcoef]
for i in range(0, 5):
    print("{:.<025} {:< 010.4e}".format(finaldata.columns[maxcoef[i]], coef[i]))


# In[70]:


predictions = LR.predict(x_test2)


# In[71]:


diff = pd.DataFrame(predictions - y_test2)


# In[72]:


diff.columns = ["diff"]


# In[73]:


diff["squared"] = diff["diff"].apply(lambda x: x*x)


# In[74]:


diff.head()


# In[75]:


np.sqrt(diff["squared"].sum()/diff.shape[0])


# # RMSE = 0.13394736650342165

# ## Linearna regresija ANALIZA REZULTATA

# In[76]:


coef_col_lr = pd.concat([pd.DataFrame(LR.coef_, columns=["coef"]), pd.DataFrame(x_train2.columns, columns=["column"])], axis=1)


# In[77]:


coef_col_lr[coef_col_lr.column.isin(["MSZoning", "LotFrontage"])]


# In[78]:


coef_col_lr["abs_coef"] = coef_col_lr.coef.apply(np.abs)


# In[79]:


coef_col_lr.sort_values("abs_coef", ascending=False)


# In[80]:


### R-squared i Adj R-squared


# In[81]:


from sklearn.metrics import r2_score


# In[82]:


r = r2_score(y_test2, predictions) # 0.896


# In[83]:


predictions.shape


# In[84]:


x_test2.shape


# In[85]:


adj_r = 1 - (1 - 0.896)*(365 - 1)/(365 - 62 - 1) # 0.875


# In[86]:


r


# In[87]:


adj_r

