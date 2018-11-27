# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:40:46 2016

@author: laurieshaw
"""

import csv, requests
import pandas as pd
import datetime as dt
import numpy as np
import scipy.stats as stats
import Elo as elo

import matplotlib.pyplot as plt

def read_season(season,div="0",fdir='/Users/andrewpuopolo/Destop/Thesis/'):
    fpath = fdir + season + '/E' + div + '.csv'
    print(fpath)
    hometeam = []
    awayteam = []
    gamedate = []
    homegoals = []
    awaygoals = []
    referee = [] # for fun
    with open(fpath, 'rU') as csvfile:
        refreader = csv.reader(csvfile, dialect = 'excel')
        refreader.next() # get rid of header
        for row in refreader: # first row is header
            if row[0]=='E'+div:
                if season in ['0203'] and div=="0":
                    dateformat='%d/%m/%Y'
                elif season in ['0304','0405'] and div=="0":
                    dateformat='%d-%m-%y'
                else:
                    dateformat='%d/%m/%y'
                gamedate.append( dt.datetime.strptime(row[1],dateformat) )
                hometeam.append(row[2])
                awayteam.append(row[3])
                homegoals.append(row[4])
                awaygoals.append(row[5])
                referee.append(row[10])
    # put into series
    d = { 'HomeTeam': pd.Series(hometeam,index=gamedate),
          'AwayTeam': pd.Series(awayteam,index=gamedate),
          'HomeGoals': pd.Series(homegoals,index=gamedate,dtype=int),
          'AwayGoals': pd.Series(awaygoals,index=gamedate,dtype=int),
          'Referee': pd.Series(referee,index=gamedate) }
    df = pd.DataFrame(d)
    df['Season'] = season
    return df

def read_season_df(season,div="0",fdir='/Users/andrewpuopolo/Destop/Thesis/',verbose=True,country="E"):
    fpath = fdir + season + '/' + country + div + '.csv'
    if verbose: print(fpath)
    if country in ['D','I','SP']:
        dateformat='%d/%m/%y'
    elif season in ['0203'] and div=="0":
        dateformat='%d/%m/%Y'
    elif (season in ['0304','0405'] and div=="0") or (season in ['0102','0203','0304','0405'] and div=="1"):
        dateformat='%d-%m-%y'
    elif season in ['0203','0304','0405'] and div in ["2","3"]:
        dateformat='%d-%m-%y'
    else:
        dateformat='%d/%m/%y'
    df = pd.read_csv(fpath,header=0,index_col=1)
    df.index = pd.to_datetime(df.index,format=dateformat)
    df['HomeGoals'] = df['FTHG']
    df['AwayGoals'] = df['FTAG']
    df['Season'] = season
    return df

def read_fixtures_df(season,div="0",fdir='/Users/andrewpuopolo/Destop/Thesis/'):
    fpath = fdir + season + '/E' + div + '_fixtures.csv'
    #print(fpath)
    dateformat = '%y-%m-%d'
    df = pd.read_csv(fpath,header=0)
    df['DATE'] = pd.to_datetime(df['DATE'],format=dateformat)
    df['Season'] = season
    return df

def read_distance_matrix():
    fdir='/Users/andrewpuopolo/Destop/Thesis/'
    fpath_clubcity = fdir + 'ClubsCityMap.csv'
    fpath_matrix = fdir + 'DistanceMatrix.csv'
    DistanceMatrix_df = pd.read_csv(fpath_matrix,index_col = 0, header=0)
    ClubCityMap = {}
    with open(fpath_clubcity, 'rU') as csvfile:
        refreader = csv.reader(csvfile, dialect = 'excel')
        refreader.next() # get rid of header
        for row in refreader:
            ClubCityMap[row[0]] = row[1]
    return (ClubCityMap, DistanceMatrix_df)
    
def read_1888results():
    path= '/Users/andrewpuopolo/Documents/Football/Data/Football-Data/ResultsSince1888.csv'
    results1888 = pd.read_csv(path, header=0, low_memory=False)
    return results1888

def read_seasons(start=1996,end=2016, div="0",verbose=True,country="E"):
    for i in range(start,end+1):
        season = get_season_from_date( dt.datetime(i,7,1) )
        results_df = read_season_df(season,div=div,verbose=verbose,country=country)
        # run a quick check
        get_teams_in_league(results_df,checkteams=True)
        if i==start:
            all_results = results_df
        else:
            all_results = pd.concat([all_results, results_df])
    return all_results
 
def get_footballdatacouk_scores(season,div="0",backup=False,country="E"):
    # This is EPL only. Set to E1,E2, etc for lower divisions.
    path = season + "/" + country + div + ".csv"
    url = "http://www.football-data.co.uk/mmz4281/" + path
    print(url)
    r = requests.get(url)
    # print output to file (for backup)
    if backup:
        backup_path = '/Users/andrewpuopolo/Destop/Thesis/' + path
        print(backup_path)
        with open(backup_path, "w") as text_file:
            text_file.write(r.text)
    # now parse the string
    lines = r.text.split('\r\n')           
    hometeam = []
    awayteam = []
    gamedate = []
    homegoals = []
    awaygoals = []
    referee = [] # for fun    
    for l in lines[1:]:
        if len(l)>18 and l[0] in ['E','D','I','SP']:
            row = l.split(',')
            gamedate.append( dt.datetime.strptime(row[1],'%d/%m/%y') )
            hometeam.append(row[2])
            awayteam.append(row[3])
            homegoals.append(row[4])
            awaygoals.append(row[5])
            referee.append(row[10])
    # put into series
    d = { 'HomeTeam': pd.Series(hometeam,index=gamedate),
          'AwayTeam': pd.Series(awayteam,index=gamedate),
          'HomeGoals': pd.Series(homegoals,index=gamedate,dtype=int),
          'AwayGoals': pd.Series(awaygoals,index=gamedate,dtype=int),
          'Referee': pd.Series(referee,index=gamedate) }
    results_df = pd.DataFrame(d)
    results_df['Season'] = season
    return results_df

def get_results_all_teams(results_df,verbose=False):
    teams = get_teams_in_league(results_df)
    results_byteam = {}
    for team in teams:
        if verbose: print(team)
        results_byteam[team] = get_results_for_team(results_df,team)
    return results_byteam

def add_RPD_to_teams(results_byteam):
    for team,games in results_byteam.items():
        ngames = len(games)
        rpd = np.ones( ngames, dtype=int )
        for i in range(ngames):
            (a,b,rpd[i]) = rolling_points_dif( games.iloc[i]['Team'], games.iloc[i]['Opponent'], results_byteam, games.index[i], ngames=38, lookback=1.5 )
        results_byteam[team]['RollingPointsDif'] = pd.Series(np.array(rpd,dtype=int),index=games.index)
    return results_byteam        
    
def get_teams_in_league(results_df, checkteams=False):
    home = set(results_df['HomeTeam'].values)
    away = set(results_df['AwayTeam'].values)
    # take union
    both = home | away
    # check for intersection is perfect, and figure out how to iterate over them
    if checkteams and len(home-away)>0:
        print (" home and away teams do not match")
        return
    return both
    
def get_season_from_date(date):
    # season moves on from July 1st
    y = date.year
    if date.month>=7: y += 1
    if y==2000: 
        return '9900'
    elif y == 2010:
        return '0910'
    elif y>2000 and y < 2010: 
        return '0' + str(np.mod(y-1,100)) + '0' + str(np.mod(y,100))
    else:
        return str(np.mod(y-1,100)) + str(np.mod(y,100))
    
def get_results_for_team(results_df, teamname):
    home = results_df[ results_df['HomeTeam'] == teamname]
    away = results_df[ results_df['AwayTeam'] == teamname]
    GoalsFor = pd.concat( [home['HomeGoals'],away['AwayGoals']] )  
    team_results_df = pd.DataFrame(GoalsFor,columns = ['GoalsFor'])
    team_results_df['GoalsAgainst'] = pd.concat( [home['AwayGoals'],away['HomeGoals']] )
    team_results_df['Opponent'] = pd.concat( [home['AwayTeam'],away['HomeTeam']] )
    if 'Referee' in home.keys():
        team_results_df['Referee'] = pd.concat( [home['Referee'],away['Referee']] )
    team_results_df['HomeAway'] = pd.concat( [pd.Series('Home',index=home.index),pd.Series('Away',index=away.index)] )
    team_results_df['Season'] = pd.concat( [home['Season'],away['Season']] )
    team_results_df['Shots'] = pd.concat( [home['HS'],away['AS']] )
    team_results_df['ShotsTarget'] = pd.concat( [home['HST'],away['AST']] )
    team_results_df['OppShots'] = pd.concat( [home['AS'],away['HS']] )
    team_results_df['OppShotsTarget'] = pd.concat( [home['AST'],away['HST']] )
    team_results_df['Fouls For'] = pd.concat( [home['HF'],away['AF']] )
    team_results_df['Fouls Against'] = pd.concat( [home['AF'],away['HF']] )
    team_results_df['Yellows For'] = pd.concat( [home['HY'],away['AY']] )
    team_results_df['Yellows Against'] = pd.concat( [home['AY'],away['HY']] )
    team_results_df['Reds For'] = pd.concat( [home['HR'],away['AR']] )
    team_results_df['Reds Against'] = pd.concat( [home['AR'],away['HR']] )
    team_results_df['Season'] = pd.concat( [home['Season'],away['Season']] )
    team_results_df['Win1']=pd.concat([home['B365H'], away['B365A']])
    team_results_df['Loss1']=pd.concat([home['B365A'], away['B365H']])
    team_results_df['Draw1']=pd.concat([home['B365D'], away['B365D']])
    team_results_df['Win2']=pd.concat([home['LBH'], away['LBA']])
    team_results_df['Loss2']=pd.concat([home['LBA'], away['LBH']])
    team_results_df['Draw2']=pd.concat([home['LBD'], away['LBD']])
    team_results_df['Win3']=pd.concat([home['IWH'], away['IWA']])
    team_results_df['Loss3']=pd.concat([home['IWA'], away['IWH']])
    team_results_df['Draw3']=pd.concat([home['IWD'], away['IWD']])
    team_results_df['Team'] = teamname
    return team_results_df.sort_index()

def build_season_table( results_byteam, tabledate, season):
    # build league table for season to tabledate
    # need to iterate through, calculating points, goal dif, goals for, name
    table = []
    teamid = 0
    for team in results_byteam.keys():
        res = results_byteam[team]
        res = res[ res['Season'] == season ]
        if len(res)>0:
            res = res[ res.index < tabledate ]
            games_played = len(res.index)
            table_Points = 3 * np.sum( res['GoalsFor']>res['GoalsAgainst'] ) + np.sum( res['GoalsFor']==res['GoalsAgainst'] )
            table_GoalFor = np.sum( res['GoalsFor'].values )
            table_GoalAga = np.sum( res['GoalsAgainst'].values )
            table_GoalDif = table_GoalFor - table_GoalAga
            table.append( (team, teamid, games_played, table_Points, table_GoalDif, table_GoalFor, table_GoalAga) )
            teamid += 1
    table = sorted(table,key = lambda team : (team[3],team[4],team[5],team[0]), reverse=True)
    return table

def team_rolling_average(team, value, window, lag):
    # make sure sorted correctly
    newheader = value + '_ave'
    team = team.sort_index(ascending=True)
    ra = np.convolve( team[value].values, np.ones(window,dtype=float)/float(window), mode='valid')
    # if lagging, pad the first lag elements
    pad = np.array( [np.nan]*(window-1+lag) )
    ra = np.concatenate( (pad,ra[:len(ra)-lag]) )
    team[newheader] =  pd.Series(ra,index=team.index)
    return team

def rolling_points_dif( team1, team2, results_byteam, gamedate, ngames=38, lookback=1.5, prom_points_per_game=1.05):
    min_startdate = gamedate - dt.timedelta(days=lookback*365)
    # get games prior to gamedate
    team1_games = results_byteam[team1][results_byteam[team1].index<gamedate]
    team2_games = results_byteam[team2][results_byteam[team2].index<gamedate]
    # select games since start date sort descening
    team1_games = team1_games[team1_games.index>=min_startdate].sort_index(ascending=False)
    team2_games = team2_games[team2_games.index>=min_startdate].sort_index(ascending=False)
    ngames_team1 = len(team1_games.index)
    ngames_team2 = len(team2_games.index)
    # snip to the correct length
    team1_games = team1_games[0:min(ngames,ngames_team1)]
    team2_games = team2_games[0:min(ngames,ngames_team2)]
    # calculate points
    team1_points = 3 * np.sum( team1_games['GoalsFor']>team1_games['GoalsAgainst'] ) + np.sum( team1_games['GoalsFor']==team1_games['GoalsAgainst'] )
    team2_points = 3 * np.sum( team2_games['GoalsFor']>team2_games['GoalsAgainst'] ) + np.sum( team2_games['GoalsFor']==team2_games['GoalsAgainst'] )    
    # correct points if team games < ngames
    team1_points += prom_points_per_game*max(ngames-ngames_team1,0)
    team2_points += prom_points_per_game*max(ngames-ngames_team2,0)
    points_dif = int(np.round(team1_points-team2_points))
    return (team1_points, team2_points, points_dif)
  
def get_distance_between_clubs(team1,team2,DistanceMatrix_df,ClubCityMap):
    return DistanceMatrix_df[ClubCityMap[team1]].loc[ClubCityMap[team2]]
    
def get_ELO_on_matchdates(team_name,team_results,elo_df,elo_change_period=5):
    matchday_elo = []
    matchday_elo_opp = []
    for r in team_results.iterrows():
        e,d = elo.get_teamELO_on_date(elo_df,team_name,r[0]-dt.timedelta(days=1))
        eopp,d = elo.get_teamELO_on_date(elo_df,r[1]['Opponent'],r[0]-dt.timedelta(days=1))
        matchday_elo.append(e)
        matchday_elo_opp.append(eopp)
    team_results['Elo'] = pd.Series(np.array(matchday_elo,dtype=float),index=team_results.index)
    team_results['EloOpp'] = pd.Series(np.array(matchday_elo_opp,dtype=float),index=team_results.index)
    team_results['Elo_change'] = team_results['Elo'].diff(elo_change_period).fillna(0)
    return team_results

def add_gameweek(results_byteam):
    teams = results_byteam.keys()
    # first do points
    for team in teams:
        res_team = results_byteam[team]
        seasons = list(set(res_team['Season'].values))
        gameweeks = pd.Series(data=None,index=None)
        for season in seasons:
            res_season = res_team[res_team['Season']==season].sort_index(ascending=True)
            ngames = len( res_season.index )
            print(ngames)
            gameweeks_season = np.arange(1,ngames+1,1)
            gameweeks = gameweeks.append( pd.Series( data=gameweeks_season, index=res_season.index ) )
        results_byteam[team]['GameWeeks'] = gameweeks.sort_index(ascending=True) 
    return results_byteam

def add_expected_goals(results_byteam):
    h = lambda x: 1 if x=='Home' else -1
    teams = results_byteam.keys()
    model = forecast.model_fit('Poisson',2)
    model.set_defaults()
    for team in teams:
        exp_goals = []
        exp_goals_opp = []
        for r in results_byteam[team].iterrows():
            matchdate = r[0]
            opp = r[1]['Opponent']
            elo_dif = (r[1]['Elo']-results_byteam[opp]['Elo'].ix[matchdate])/400.
            X = np.array([elo_dif,h(r[1]['HomeAway'])])
            model.generate_prediction(X,-X)
            exp_goals.append(model.mu1)
            exp_goals_opp.append(model.mu2)
        results_byteam[team]['ExpGoals'] = pd.Series(np.array(exp_goals),index=results_byteam[team].index)
        results_byteam[team]['OppExpGoals'] = pd.Series(np.array(exp_goals_opp),index=results_byteam[team].index)
    return results_byteam
        
def calc_season_points(results_byteam):
    p = lambda x: 3 if x>0 else 0 if x<0 else 1
    teams = results_byteam.keys()
    # first do points
    for team in teams:
        res_team = results_byteam[team]
        seasons = list(set(res_team['Season'].values))
        points = pd.Series(data=None,index=None)
        for season in seasons:
            res_season = res_team[res_team['Season']==season].sort_index(ascending=True)
            goaldif = res_season['GoalsFor'].values - res_season['GoalsAgainst'].values 
            point_season = np.cumsum( [p(gd) for gd in goaldif] )
            points = points.append( pd.Series( data=point_season, index=res_season.index ) )
        results_byteam[team]['Points'] = points.sort_index(ascending=True)    
    return results_byteam
            
def calc_runs(results_byteam):
    win = lambda x: 1 if x>0 else 0
    unb = lambda x: 1 if x>=0 else 0
    teams = results_byteam.keys()
    # first do points
    for team in teams:
        res_team = results_byteam[team]
        seasons = list(set(res_team['Season'].values))
        winning_runs = pd.Series(data=None,index=None)
        unbeaten_runs = pd.Series(data=None,index=None)
        for season in seasons:
            res_season = res_team[res_team['Season']==season].sort_index(ascending=True)
            goaldif = res_season['GoalsFor'].values - res_season['GoalsAgainst'].values
            unbeaten = np.zeros(len(goaldif),dtype=int)
            winning = np.zeros(len(goaldif),dtype=int)
            unbeaten[0] = unb(goaldif[0])
            winning[0] = win(goaldif[0])
            for i in np.arange(1,len(goaldif)):
                if goaldif[i]>0:
                    unbeaten[i] = unbeaten[i-1]+1
                    winning[i] = winning[i-1]+1
                elif goaldif[i]==0:
                    unbeaten[i] = unbeaten[i-1]+1
                    winning[i] = 0
                else:
                    winning[i] = 0
                    unbeaten[i] = 0
            winning_runs = winning_runs.append( pd.Series( data=winning, index=res_season.index ) )
            unbeaten_runs = unbeaten_runs.append( pd.Series( data=unbeaten, index=res_season.index ) )
        results_byteam[team]['Winning'] = winning_runs.sort_index(ascending=True) 
        results_byteam[team]['Unbeaten'] = unbeaten_runs.sort_index(ascending=True) 
    return results_byteam
            
def make_goaldif_plot(results_byteam):
    years = np.arange(1995,2016,1)
    results = {'goaldiff':[],'for':[],'against':[],'points':[],'position':[],'seasons':[]}
    seasons = []
    tables = []
    for year in years:
        seasons.append(str(year)[2:] + str(year+1)[2:])
        tables.append(build_season_table( results_byteam, dt.datetime(2018,6,1), seasons[-1]))
    for table in tables:
        x  = np.array([t[5]-t[4] for t in table])
        results['against'] =  np.concatenate( [results['against'],x] )
        x  = np.array([t[3] for t in table])
        results['points'] =  np.concatenate( [results['points'],x] )
        x  = np.array([t[-1] for t in table])
        results['for'] =  np.concatenate( [results['for'],x] )
        x  = np.array([t[4] for t in table])
        results['goaldiff'] =  np.concatenate( [results['goaldiff'],x] )
        results['position'] =  np.concatenate( [results['position'], np.arange(1,21,1)] )
    fig,ax = plt.subplots()
    ax.plot(results['for']/38.,results['against']/38., 'k+')
    cut = results['position']<=4
    ax.plot(results['for'][cut]/38.,results['against'][cut]/38., 'bo')
    cut = results['position']==1
    ax.plot(results['for'][cut]/38.,results['against'][cut]/38., 'go')
    cut = results['position']>=18
    ax.plot(results['for'][cut]/38.,results['against'][cut]/38., 'ro')
    ax.plot([1.0,3.0],[0.0,2.0],'k--')
    ax.plot([0.0,2.5],[0.5,3.0],'k--')
    ax.plot([0.5,3.0],[0,2.5],'k--')
    ax.yaxis.grid()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    