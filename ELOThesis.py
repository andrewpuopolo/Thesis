#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:42:03 2018

@author: andrewpuopolo
"""

import pandas as pd
import datetime as dt
import footballData as fd
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import time
import Elo as ELO

df1888=pd.read_csv('engsoccerdata2.csv')

Post4thSplit=df1888[df1888['Season']>1982.5]
Post4thSplit=Post4thSplit.sort_values('Date')
Post4thSplit=Post4thSplit.reset_index()
del Post4thSplit['index']


 
#Making Wimbledon FC and Milton Keynes Dons 1 club
for i in range(len(Post4thSplit)):
    if(Post4thSplit['Season'][i]>2003.5 and Post4thSplit['home'][i]=='Milton Keynes Dons'):
        Post4thSplit['home'][i]='Wimbledon'
    if(Post4thSplit['Season'][i]>2003.5 and Post4thSplit['visitor'][i]=='Milton Keynes Dons'):
        Post4thSplit['visitor'][i]='Wimbledon'
        
#Get initial teams and their divisions
teams1983=Post4thSplit[Post4thSplit['Season']==1983]

initialrows=[]
for i in range(len(teams1983['home'].unique())):
    Team=teams1983['home'].unique()[i]
    Teamdf=teams1983[teams1983['home']==Team].reset_index()
    Tier=int(Teamdf['tier'][0])
    initialrow=[Team, Tier]
    initialrows.append(initialrow)
print(initialrows)

Elodictionary={}
initvalues=[1750,1500,1250,1000]
HFA=80
k=20
L2Pro=950

for a in range(len(initialrows)):
    Elodictionary[initialrows[a][0]]=initvalues[initialrows[a][1]-1]


time1=time.time()

briersums=[]
logliksums=[]
burnin=[]
ks=[]
for b in range(14):
    for k in range(15,30,1):
        print([k, b])
        Briers=[]
        logliks=[]
        BurnIn=5
        Elodictionary={}
        initvalues=[1750,1500,1250,1000]
        HFA=75
        L2Pro=950
        for a in range(len(initialrows)):
            Elodictionary[initialrows[a][0]]=initvalues[initialrows[a][1]-1]
        for j in range(len(Post4thSplit)):
            Home=Post4thSplit['home'][j]
            Away=Post4thSplit['visitor'][j]
            Division=Post4thSplit['tier'][j]-1
            if Home in Elodictionary:
                HomeELO=Elodictionary[Home]
            else:
                Elodictionary[Home]=L2Pro
                HomeELO=Elodictionary[Home]
            if Away in Elodictionary:
                AwayELO=Elodictionary[Away]
            else:
                Elodictionary[Away]=L2Pro
                AwayELO=Elodictionary[Away]
            Elodif=float((HomeELO-AwayELO+HFA)/400)
            E=1/(10**(-1*(Elodif))+1)
            if Post4thSplit['hgoal'][j]>Post4thSplit['vgoal'][j]:
                R=1
                MatchLogLik=np.log(E)
            elif Post4thSplit['hgoal'][j]<Post4thSplit['vgoal'][j]:
                R=0
                MatchLogLik=np.log(1-E)
            else:
                R=.5
                MatchLogLik=np.log(np.sqrt(E*(1-E)))
            PointExchange=(R-E)*k
            Elodictionary[Home]=HomeELO+PointExchange
            Elodictionary[Away]=AwayELO-PointExchange
            if Post4thSplit['Season'][j]>(1982+b):
                logliks.append(MatchLogLik)
        logliksums.append(np.sum(logliks))
        ks.append(k)
        burnin.append(b)
        


   
     
#ELOlist=[]
#columns=['Team', 'Rating']
#for i in Elodictionary:
#    ELOlist.append([i, Elodictionary[i]])
#x=pd.DataFrame(ELOlist, columns=columns)
#y=x.sort_values('Rating')
#print(y)

rows=[]
cols=['k', 'burn in','loglik']
for i in range(len(ks)):
    rows.append([ks[i], burnin[i], logliksums[i]])
plottingdf=pd.DataFrame(rows, columns=cols)


minkrow=[]
cols2=['k', 'Min LogLik']
for i in range(14):
    newdf=plottingdf[plottingdf['burn in']==i].reset_index()
    loglikmin=newdf.loc[newdf['loglik'].idxmax()]
    print(loglikmin['burn in'], loglikmin['k'])
    minkrow.append([loglikmin['burn in'], loglikmin['k']])

minks=pd.DataFrame(minkrow,columns=cols2)

minks['Min LogLik'].plot(kind='line', color='red', label='Log Likelihood', legend=True)
plt.xlabel('Burn In Period')
plt.ylabel('Optimal k')
plt.title('Optimal k by Burn In Period Using Log Likelihood and Brier as Inputs')
plt.show()


for j in range(14):
    newdf=plottingdf[plottingdf['burn in']==j].reset_index()
    x=newdf['k']
    y=newdf['loglik']
    plt.plot(x, y)
    plt.xlabel('K')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood/Brier by k with Burn In Period Of ' + str(j) + ' Years')
    plt.show()
    
    
newdf=plottingdf[plottingdf['burn in']==5].reset_index()
newdf['loglik'].plot(kind='line', color='red', legend=True)
plt.xlabel('K')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood/Brier by k with Burn In Period Of 5 Years')
plt.show()








Post4thSplit=df1888[df1888['Season']>1957.5]
Post4thSplit=Post4thSplit.sort_values('Date')
Post4thSplit=Post4thSplit.reset_index()
del Post4thSplit['index']


 
#Making Wimbledon FC and Milton Keynes Dons 1 club
for i in range(len(Post4thSplit)):
    if(Post4thSplit['Season'][i]>2003.5 and Post4thSplit['home'][i]=='Milton Keynes Dons'):
        Post4thSplit['home'][i]='Wimbledon'
    if(Post4thSplit['Season'][i]>2003.5 and Post4thSplit['visitor'][i]=='Milton Keynes Dons'):
        Post4thSplit['visitor'][i]='Wimbledon'
        
#Get initial teams and their divisions
teams1983=Post4thSplit[Post4thSplit['Season']==1958]

initialrows=[]
for i in range(len(teams1983['home'].unique())):
    Team=teams1983['home'].unique()[i]
    Teamdf=teams1983[teams1983['home']==Team].reset_index()
    Tier=int(Teamdf['tier'][0])
    initialrow=[Team, Tier]
    initialrows.append(initialrow)
print(initialrows)


briersums2=[]
logliksums2=[]
hvals=[]
ks2=[]
hs=[40,45,50,55,60,65,70,75,80,85,90,95,100]
for h in hs:
    for k in range(15,30,1):
        print([k, h])
        Briers=[]
        logliks=[]
        Elodictionary={}
        initvalues=[1750,1500,1250,1000]
        HFA=75
        L2Pro=950
        for a in range(len(initialrows)):
            Elodictionary[initialrows[a][0]]=initvalues[initialrows[a][1]-1]
        for j in range(len(Post4thSplit)):
            Home=Post4thSplit['home'][j]
            Away=Post4thSplit['visitor'][j]
            Division=Post4thSplit['tier'][j]-1
            if Home in Elodictionary:
                HomeELO=Elodictionary[Home]
            else:
                Elodictionary[Home]=L2Pro
                HomeELO=Elodictionary[Home]
            if Away in Elodictionary:
                AwayELO=Elodictionary[Away]
            else:
                Elodictionary[Away]=L2Pro
                AwayELO=Elodictionary[Away]
            Elodif=float((HomeELO-AwayELO+h)/400)
            E=1/(10**(-1*(Elodif))+1)
            if Post4thSplit['hgoal'][j]>Post4thSplit['vgoal'][j]:
                R=1
                MatchLogLik=np.log(E)
            elif Post4thSplit['hgoal'][j]<Post4thSplit['vgoal'][j]:
                R=0
                MatchLogLik=np.log(1-E)
            else:
                R=.5
                MatchLogLik=np.log(np.sqrt(E*(1-E)))
            PointExchange=(R-E)*k
            Elodictionary[Home]=HomeELO+PointExchange
            Elodictionary[Away]=AwayELO-PointExchange
            if Post4thSplit['Season'][j]>(1982):
                logliks.append(MatchLogLik)
        logliksums2.append(np.sum(logliks))
        hvals.append(h)
        ks2.append(k)

rows=[]
cols=['k', 'Home Field','loglik']
for i in range(len(ks)):
    rows.append([ks[i], hvals[i], logliksums2[i]])
plottingdf2=pd.DataFrame(rows, columns=cols)

globalminimum=plottingdf2.loc[plottingdf2['loglik'].idxmax()]
print(globalminimum)


for j in range(15,30,1):
    newdf=plottingdf2[plottingdf2['k']==j].reset_index()
    loglikmin=newdf.loc[newdf['loglik'].idxmax()]
    print([j, loglikmin['Home Field']])
    x=newdf['Home Field']
    y=newdf['loglik']
    plt.plot(x, y)
    plt.xlabel('K')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood/Brier by k value of  ' + str(j))
    plt.show()
    

plt.contour([plottingdf2['Home Field'], plottingdf2['k']] plottingdf2['loglik'])
