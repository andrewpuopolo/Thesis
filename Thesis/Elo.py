7# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:10:12 2016

@author: laurieshaw
"""

import csv, requests
import pandas as pd
import datetime as dt
import numpy as np
from bs4 import BeautifulSoup
import re 
import unicodedata
import string

def get_ELO_for_club(clubname):
    elo_dict = build_elo_dict()
    if clubname in elo_dict.keys():
        clubname = elo_dict[clubname]
    url = "http://api.clubelo.com/" + clubname
    print(url)
    r = requests.get(url)
    # now parse the string
    lines = r.content.split('\n')  
    if len(lines)<5:
        print("no data for %s" % clubname)         
    club = []
    score = []
    country = []
    date_from = []
    date_to = []
    for l in lines[1:]:
        if len(l)>0:
            row = l.split(',')
            date_from.append( dt.datetime.strptime(row[5],'%Y-%m-%d') )
            date_to.append( dt.datetime.strptime(row[6],'%Y-%m-%d') )
            club.append(row[1])
            score.append(float(row[4]))
            country.append(row[2])
    # put into series
    elo_club =  pd.Series(score,index=date_from)
    return elo_club

def get_teamELO_on_date(elo_df,team,date):
    i = elo_df[team].index.get_loc(date,method='ffill')
    if elo_df[team].index[i]>date: print ("ELO date error")
    return float(elo_df[team].values[i]),elo_df[team].index[i]

def get_max_ELOdiff_teams(elo_df,teams,date):
    elos = []
    for team in teams:
        e,d = get_teamELO_on_date(elo_df,team,date)
        elos.append(e)
    return max(elos)-min(elos)

def get_all_ELO(results_all_EPL):
    teams = list(set(list(results_all_EPL['HomeTeam'])+list(results_all_EPL['AwayTeam'])))
    all_elo = {}
    for team in teams:
        all_elo[team] = get_ELO_for_club(team)
    elo_df = pd.DataFrame(all_elo).fillna(method='ffill')
    return elo_df
   
def read_elo_data(fname='EloAll.csv'):
    fdir = '/Users/laurieshaw/Documents/Football/Data/Elo/' + fname
    elo_df = pd.read_csv(fdir,header=0,index_col=0)
    elo_df.index = pd.to_datetime(elo_df.index,format='%Y-%m-%d')
    return elo_df
   
def save_elo_data(elo_df,fname):
    fdir = '/Users/laurieshaw/Documents/Football/Data/Elo/' + fname
    elo_df.to_csv(path_or_buf=fdir)

def build_elo_dict():
    elo_dict = {'Man City':'ManCity',
                'Man United':'ManUnited',
                'West Ham':'WestHam',
                'West Brom':'WestBrom',
                'Aston Villa':'AstonVilla',
                'Crystal Palace':'CrystalPalace',
                'Sheffield Weds':'SheffieldWeds',
                'Sheffield United':'SheffieldUnited',
                "Nott'm Forest":"Forest",
                "Rapid Wien":"RapidWien",
                "Wiener SC":"WienerSC",
                "St.Pauli":"StPauli",
                "Munchen60":"Muenchen60",
                "Koln":"Koeln",
                "Nurnberg":"Nuernberg",
                "Dusseldorf":"Duesseldorf",
                "Furth":"Fuerth",
                "Zurich":"Zuerich",
                "Saarbrucken":"Saarbruecken",
                "RealSociedad":"Sociedad",
                "St.Gallen":"StGallen",
                "Orebro":"Oerebro",
                "Malmo":"Malmoe",
                "FCKbenhavn":"FCKobenhavn",
                "Gyor":"Gyoer",
                "Vasas":"VasasSC",
                "Goteborg":"Goeteborg"}
    return elo_dict

def replace_day_text(datestring):
    pat = ['st','nd','rd']
    for p in pat:
        datestring = re.sub(p,'th',datestring)
    return datestring

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

def QueryManagerClublist(managername):
    managername = managername.replace(" ", "")
    print(managername)
    url = "http://clubelo.com/" + managername
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    tabledata = soup.find_all('table','ranking')[1].find_all('td','r')
    clubdata = soup.find_all('table','ranking')[1].find_all('a')
    #print "nlines %s, nposts %s" % (len(clubdata),len(clubdata)/3.)
    clublist = []
    for i in np.arange(0,len(clubdata),3):
        clubname = remove_accents( clubdata[i].get_text() ).replace(" ", "")
        datefrom = dt.datetime.strptime(replace_day_text(clubdata[i+1].get_text()),'%a, %b %dth, %Y')
        dateto = dt.datetime.strptime(replace_day_text(clubdata[i+2].get_text()),'%a, %b %dth, %Y')
        if clubname:
            clublist.append((clubname, datefrom, dateto))
    return clublist[::-1]
    
