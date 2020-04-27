# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:08:57 2019

Change path in line 21 to the path of the data files !

Begles DATA READ
Assume only one xls file in the directory

@author: gruck
"""

#%% Import relevant libraries for Begles file

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

sns.set()               #preferred visualisation style
sns.set_style('ticks')  

#%% Define functions used

def jour_fer(date,df):
    """
    Input: 
        date of Jour férié in string form 'mm-jj'
        dataframe containing all the data(pandas array)
    Output:
        The input dataframe is altered in the function
        In python this is possible without global variables, no need to return values
        
    NOTE:: This assumes the data is in ten minute intervals (creates 24*6 points for one day)
    """
    timestep = 10                       #minutes
    start = df[df['datetime'] == '2018-{} 00:00:00'.format(date)].index.values.astype(int)[0] 
    end = int(start + 24*(60/timestep))
    
    df['weekday'][start:end] = 6        #relabels the weekday of a jour férié as a sunday (6)


def read_file(path,filename,sheet_name = None, delimiter = None):
    """
    Returns dataframe from the inputs given$
    Should be relatively simple 
    """
    os.chdir(path)
    filetype = file.split('.')[1]       #splits filename by delimeter '.' to give 'csv' 'xls' or 'xlsx'
    
    if filetype == 'csv':
        if delimiter == None: delimiter = ','   #csv default delimeter should be a comma, but may often be a semi-colon!
        df = pd.read_csv(file,delimiter)
        
    else:
        df = pd.read_excel(file,sheet_name)
        
    return df

def merge_arrays(df_left,df_right,method = 'left', join_column = 'datetime'):
    """
    df_left is the dataframe to add data to, df_right is the data to be added
    method: 
        'left' : use all the values of datetime in df_left regardless if they exist in df_right
        'right' : use all the values of datetime in df_right (temperature array) NOT ADVISED
        'union' : use all values in datetime from the full set of both dataframes
    Documentation for df merge is good explanation for more details.
    """
    df = df_left.merge(df_right,how = method,left_on = join_column,right_on = join_column)
    return df

#%% Main code
    
if __name__ == "__main__":
    """
    create a main code with the path as an input!!
    """
    sns.set()
    sns.set_context('talk')
    #%% locate and read data file
    path = r'C:\Users\gruck\Documents\Begles' #à modifier 
    file = 'VILLE DE BEGLES_HOTEL DE VILLE_RAE_30001611162862_CDC_22_09_2018_au_21_12_2018.xlsx'
    df = read_file(path,file,sheet_name = 'HOTEL DE VILLE_Pts_10')

    #rename the column currently labelled JJ/MM/AAAA HH:MM:SS as 'datetime' (the first column in the list of columns)
    df.rename(columns = {list(df)[0] : 'datetime'},inplace = True)
    
    #read second file with temperature data
    df_temp = pd.read_csv('donnees_temp_bordeaux_2018.csv',delimiter = ';')
    #%%change the date column from strings to recognised datetime object
    df_temp['Date'] =  pd.to_datetime(df_temp['Date'], format='%d/%m/%Y %H:%M') #format can be customised
    df_temp.sort_values(by=['Date'],inplace = True)                             #dates were not in the correct order
    df_temp.rename(columns = {'Date' : 'datetime'},inplace = True)              #same column name for both dataframes
    
    
    #%% store original array as original in case reference is necessary
    
    df_original = df.set_index('datetime') #index of original dataframe is the date
    
    #%%
    
    #join the two arrays
    df = merge_arrays(df,df_temp)
    df['Temperature'].interpolate(inplace = True)
    df.dropna(inplace = True)                       #only occurs at brief intervals at beginning and end of dataset
    
    
    #%% plot four subplots to have an idea of the data
    
    #add column for weekday 0(monday) - 6(sunday)
    df['weekday'] = df['datetime'].dt.dayofweek #reliant on datetime format
    
    #jours férié - jeudi 1 NOV (only one jour férié in training set)
    jour_fer('11-01',df)
    
    df['heure'] = df['datetime'].dt.hour                                    #column for hour
    df['heure minutes'] = df['datetime'].dt.strftime('%H:%M')               #column for hour minutes
    df['month'] = df['datetime'].dt.month
    
    df['echauffement'] = df['datetime']>datetime.datetime(2018,11,20,0,0)   #périod d'échauffe (commencant le 20 novembre)
    df.set_index('datetime', inplace = True)                                #set index as date after extra columns formed
    
    #heure minutes (hm) as a continuous scale
    df['hm'] = df['heure minutes'].apply(lambda x: int(100*int(x.split(':')[0])+(5/3)*int(x.split(':')[1])))
    
    #save file
    df.to_csv(r'begles_3month_Data.csv',sep = ';')
    
    #%% plots
    
    fig_sig,ax_sig = plt.subplots(2,2,figsize = (18,10)) 
    plt.suptitle('Données 01/09 - 20/12 2018')
    
    ax_sig[0,0].plot(df_original)
    ax_sig[0,0].set_title('Courbe de Charge')
    ax_sig[0,0].set_ylabel('P / kW',rotation = 90)
    ax_sig[0,0].xaxis.set_major_locator(plt.LinearLocator(4))
    
    ax_sig[0,1].plot(df.groupby('heure minutes').mean()['PS ATTEINTE'])
    ax_sig[0,1].set_title('PS ATTEINTE Moyenne Journalière')
    ax_sig[0,1].set_ylabel('P / kW',rotation = 90)
    ax_sig[0,1].xaxis.set_major_locator(plt.LinearLocator(9))
    
    ax_sig[1,0].plot(df.groupby('weekday').mean()['PS ATTEINTE'])
    ax_sig[1,0].set_title('PS ATTEINTE Moyenne Semaine')
    ax_sig[1,0].set_ylabel('P / kW',rotation = 90)
    ax_sig[1,0].set_xlabel('Jour de la Semaine')
    
    ax_sig[1,1].scatter(df['Temperature'],df['PS ATTEINTE'])
    ax_sig[1,1].set_title('Distribution PS ATTEINTE - Temperature')
    ax_sig[1,1].set_ylabel('P / kW',rotation = 90)
    
    #tight layout
    fig_sig.tight_layout(rect = [0, 0.03, 1, 0.95])
    
    #fig_sig.savefig('firstPlots.png')
    
    #%% output df - creat a df with the columns minus first two for the unknown power months that we want to predict
    
    numdays,numdays_end = 243,9
    date_list = [df.index[0] - datetime.timedelta(minutes = x) - datetime.timedelta(hours = 2) for x in np.linspace(0,numdays*24*6*10,numdays*24*6+1)][::-1]
    date_list_end = [df.index[-1] + datetime.timedelta(minutes = x) for x in np.linspace(10,numdays_end*24*6*10+10,numdays_end*24*6+1)]
    
    #create dataframe with dates for up to end of august as values
    
    df_pred = pd.DataFrame(np.hstack((date_list,date_list_end)),
                           columns = ['datetime'])
    df_pred = df_pred.merge(df_temp, how='left', left_on='datetime', right_on='datetime')
    
    df_pred['Temperature'].interpolate(inplace = True)
    
    #add column for weekday
    df_pred['weekday'] = df_pred['datetime'].dt.dayofweek
    
    #jours férié - 
    joursferies = '01-01 04-02 05-01 05-08 05-10 05-22 07-14 08-15 12-25'.split()
    for jour in joursferies:
        jour_fer(jour,df_pred)
    
    df_pred.dropna(inplace = True)
    
    df_pred['heure'] = df_pred['datetime'].dt.hour
    df_pred['heure minutes'] = df_pred['datetime'].dt.strftime('%H:%M')
    df_pred['month'] = df_pred['datetime'].dt.month
    
    df_pred['echauffement'] = (df_pred['datetime']<datetime.datetime(2018,4,20,0,0))|(df_pred['datetime']>datetime.datetime(2018,11,20,0,0))
    
    #minutes en decimal de l'heure (fraction of an hour)
    df_pred['hm'] = df_pred['heure minutes'].apply(lambda x: int(100*int(x.split(':')[0])+(5/3)*int(x.split(':')[1])))
    
    df_pred.sort_values(by=['datetime'],inplace = True)
    df_pred.set_index('datetime', inplace = True)
    
    
    #%% arbre decisionel
    
    from sklearn.tree import DecisionTreeRegressor
    #from sklearn.tree import export_graphviz
    
    from sklearn.ensemble import AdaBoostRegressor
    
    
    X = df[['Temperature','weekday','echauffement','hm']]
    Y = df['PS ATTEINTE']
    
    tree = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15), n_estimators=20)
    tree.fit(X,Y)
    
    #export_graphviz(tree, out_file = 'tree.txt')
    
    #%%
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33)
    
    scores = []
    for i in range(1,20):
        tree1 = DecisionTreeRegressor(max_depth=i)
        tree1.fit(X_train,Y_train)
        
        valid_pred = tree1.predict(X_test)
        scores.append(tree1.score(X_test,Y_test))
    
    #%%
    
    train_pred = tree.predict(X)
    test_pred = tree.predict(df_pred[['Temperature','weekday','echauffement','hm']])
    #for underestimate multiply by empiric factor
    
    #%% PLOTS
    
    fig,ax = plt.subplots(2,1)
    fig.suptitle('September to December')
    ax[0].plot(df['PS ATTEINTE'])
    ax[1].plot(train_pred)
    
    plt.figure()
    plt.plot(test_pred)
    
    #%% Final Plot
    
    plt.figure()
    plt.plot(df_pred.index,test_pred,color = 'dodgerblue')
    plt.plot(df.index,Y,color = 'blue')
    plt.title('Prévision de Puissance Atteinte - 2018')
    plt.xlabel('Date')
    plt.ylabel(r'P / $kW$')
    
    #%% Create full data set
    
    df_pred['PS ATTEINTE'] = test_pred
    data2018 = pd.DataFrame(pd.concat([df_pred,df]))
    data2018.sort_index(inplace = True)
    
    #%% VALIDATION
    factures = [38834,49710,45601,38196,20017,15536,16726,13784,17302,11769,25547,40227]
    df_ps = data2018.reset_index()
    df_ps.set_index('month',inplace = True)
    energy_use = []
    for i in range(1,13): energy_use.append(np.trapz(df_ps.loc[i]['PS ATTEINTE'])//6)
    
    df_validation = pd.DataFrame({'factures':factures,'previsions':energy_use})
    df_validation['month'] = 'J F M A M J J A S O N D'.split()
    
    df_validation.plot(x = 'month', y = ['factures','previsions'], kind = 'bar')
    plt.ylabel('kWh facturés')
    plt.title('Consommation calculée vs Consommation calculée')
    
    correlations = {'annee':df_validation.corr(),'donnees':df_validation.iloc[8:].corr()}
    
    print('SUM of FACTURES: {},\n Sum of Energy USE: {}'.format(sum(factures),sum(energy_use)))