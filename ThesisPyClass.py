
# coding: utf-8

# In[53]:

import pandas as pd
import datetime as dt
import footballData as fd
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import time
import Elo as ELO
import seaborn as sns


# In[54]:

df1888=pd.read_csv('engsoccerdata2.csv')
df1888


# In[55]:

startdate=1983
fulldf=df1888[df1888['Season']>=startdate]
fulldf=fulldf.sort_values('Date')
fulldf=fulldf.reset_index()
fulldf=fulldf.replace('Milton Keynes Dons', 'Wimbledon')
fulldf


# In[56]:

initdictionary={}
initvalues=[1750,1500,1250,1000]
initialrows=[]
#This needs to be rewritten so we don't have to run it over an dover again
for i in range(len(fulldf['home'].unique())):
    Team=fulldf['home'].unique()[i]
    Teamdf=fulldf[fulldf['home']==Team].reset_index()
    Tier=int(Teamdf['tier'][0])
    initialrow=[Team, Tier]
    initialrows.append(initialrow)
#Needs to be rewritten to optimize using BuildEloDict Function
for a in range(len(initialrows)):
    initdictionary[initialrows[a][0]]=initvalues[initialrows[a][1]-1]





# In[35]:

initdictionary


# In[151]:

def loglik(k, initHF):
    HFA=initHF
    logliks=0.0
    burn_in=5
    Elodictionary=initdictionary.copy()
    startyear=fulldf['Season'].values[0]
    HFAyear=fulldf['Season'].values[0]
    HFcounter=0.
    Gamecounter=0.
    for ind, row in fulldf.iterrows():
        Home=row['home']
        Away=row['visitor']
        HomeGoals=row['hgoal']
        AwayGoals=row['vgoal']
        year=row['Season']
        HomeELO=Elodictionary[Home]
        AwayELO=Elodictionary[Away]
        Elodif=float((HomeELO-AwayELO+HFA)/400.0)
        E=1/(10**(-1*(Elodif))+1)
        #Write Lambda Functions for this to calculate Rvalue and Loglik, gets rid of if statements
        Resfun = lambda x,y: (x > y)*1 + (x == y)*(.5)
        R=Resfun(HomeGoals, AwayGoals)
        mloglikfun = lambda x,y: np.log(np.sqrt(y*(1-y))) if x==.5 else np.log((-1*x)+1+(2*x-1)*y)
        MatchLogLik= mloglikfun(R, E)
        #if HomeGoals>AwayGoals:
         #   R=1
          #  MatchLogLik=np.log(E)
       # elif HomeGoals<AwayGoals:
        #    R=0
         #   MatchLogLik=np.log(1-E)
       # else:
        #    R=.5
         #   MatchLogLik=np.log(np.sqrt(E*(1-E)))
        PointExchange=(R-E)*k
        Elodictionary[Home]=HomeELO+PointExchange
        Elodictionary[Away]=AwayELO-PointExchange
        HFcounter+=(R-E)
        Gamecounter+=1.
        if year>(startyear+burn_in):
            #print(HFcounter)
            logliks+=MatchLogLik
        #if year>(HFAyear) and year >(startyear+burn_in):
            #print(HFcounter)
         #   HFA+=(HFcounter/Gamecounter)*Hk
          #  HFAyear=year
           # HFcounter=0
            #Gamecounter=0
    #print([k, initHF, np.sum(logliks)])
    #print(Elodictionary)
    return -1*logliks


# In[157]:

gridsearch=[]

for a in range(10,40,1):
    for b in range(40,100,4):
        l=loglik(a,b)
        z=[a,b,-1*l]
        print(z)
        gridsearch.append(z)


# In[169]:

np.random.uniform(0,1000)


# In[172]:

from scipy.optimize import minimize
mins=[]
for i in range(50):
    initk=np.random.uniform(0,300)
    inithf=np.random.uniform(0,1000)
    x0=[initk,inithf]
    res=minimize(lambda x: loglik(*x),x0, method='SLSQP', bounds=[[0,300], [0,1000]],
                 options={'disp': True})
    row=[initk, inithf, res['x'][0], res['x'][1]]
    print(row)
    mins.append(row)


# In[170]:

res['x'][0]


# In[ ]:

#Build EloDict Function
#Write function so I don't have to do the unique teams a bunch of times


# In[101]:

#Learn Lambda Functions to get rid of all these if statements


# In[ ]:

#For next time, tidy up code write functions for things that can be functionized
#Run Grid Search for k and fixed HFA, map out paramater space, then randomly start within fairly broad bounds, 
#and plot optimizer value where it ends up for each of the things
R = lambda 


# In[174]:

#To debug home field advantage paramter, see what it looks like season to season
#Do we want to average, do we want to sum?
#Write out math and see what it should look like in a system
mins


# In[175]:

cols=['initK', 'initHF', 'optk', 'optHF']
minsdf=pd.DataFrame(mins, columns=cols)
minsdf


# In[204]:

cols2=['k', 'HFA', 'loglik']
gridsearchdf=pd.DataFrame(gridsearch, columns=cols2)


# In[213]:

plots = gridsearchdf.pivot("HFA", "k", "loglik")
sns.heatmap(plots, cmap="YlGnBu")
plt.xlabel('k value')
plt.ylabel('Home Field Advantage')
plt.title('Heat Map With Burn In Period Of 5 Years')
plt.show()


# In[217]:

plt.scatter( x=minsdf['optk'],y= minsdf['optHF'])
plt.show()


# In[ ]:



