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

def periode_echau(df,periode):
    if periode['start'] == None:
        df['echauffement'] = df['datetime'] < periode['end']
    elif periode['end'] == None:
        df['echauffement'] = df['datetime'] > periode['start']
    else:
        df['echauffement'] = (df['datetime'] > periode['start']) & (df['datetime'] < periode['end'])


def main(df,df_temp,startdate_pred,jours_ferie = None,periode_echauffe = None):
    
    #rename the column currently labelled JJ/MM/AAAA HH:MM:SS as 'datetime' (the first column in the list of columns)
    df.rename(columns = {list(df)[0] : 'datetime'},inplace = True)
    df_temp.rename(columns = {'Date' : 'datetime'},inplace = True)          #same column name for both dataframes
    
    #join the two arrays
    df = merge_arrays(df,df_temp)
    df['Temperature'].interpolate(inplace = True)
    df.dropna(inplace = True)                                               #only occurs at brief intervals at beginning and end of dataset
    
    #add column for weekday 0(monday) - 6(sunday)
    df['weekday'] = df['datetime'].dt.dayofweek                             #reliant on datetime format
    
    #jours férié added
    if jours_ferie != None:
        for jour in jours_ferie:
            jour_fer(jour,df)
    
    df['heure'] = df['datetime'].dt.hour                                    #column for hour
    df['heure minutes'] = df['datetime'].dt.strftime('%H:%M')               #column for hour minutes
    df['month'] = df['datetime'].dt.month                                   #column for month
    
    if periode_echauffe != None:
        periode_echau(df,periode_echauffe)                                  #if necessary add heating period
    
    df.set_index('datetime', inplace = True)                                #set index as date after extra columns formed
    
    #heure minutes (hm) as a continuous scale
    df['hm'] = df['heure minutes'].apply(lambda x: int(100*int(x.split(':')[0])+(5/3)*int(x.split(':')[1])))
    
    #save file
    df.to_csv(r'begles_3month_Data.csv',sep = ';')
    
    #save file for prediction as well
    #should then be able to run the main only for this file
    date_list = [startdate_pred + datetime.timedelta(minutes = 10*i) for i in range(365*24*6)]
    date_list = pd.Series([date_obj.strftime('%Y-%m-%d %H:%M:%S') for date_obj in date_list])
    date_list = pd.to_datetime(date_list, format='%Y-%m-%d %H:%M:%S')
    train_dates = list(df.index)
    date_list = [date for date in date_list if date not in train_dates]
    df_pred = pd.DataFrame({'datetime':date_list})
    df_pred = merge_arrays(df_pred,df_temp)
    df_pred['Temperature'].interpolate(inplace = True)
    df_pred['weekday'] = df_pred['datetime'].dt.dayofweek
    df_pred['heure'] = df_pred['datetime'].dt.hour
    df_pred['heure minutes'] = df_pred['datetime'].dt.strftime('%H:%M')
    df_pred['month'] = df_pred['datetime'].dt.month
    df_pred['hm'] = df_pred['heure minutes'].apply(lambda x: int(100*int(x.split(':')[0])+(5/3)*int(x.split(':')[1])))
    periode_echau(df_pred,periode = {'start':None,'end':datetime.datetime(2018,4,20,0,0)})
    periode_echau(df_pred,periode = {'start':datetime.datetime(2018,11,20,0,0),'end':None})
    df_pred.set_index('datetime', inplace = True)
    df_pred.to_csv(r'begles_prediction_Data.csv',sep = ';')

#%% Main code
    
if __name__ == "__main__":
    
    #locate file
    path = r'C:\Users\gruck\Documents\Begles' #à modifier 
    file = 'VILLE DE BEGLES_HOTEL DE VILLE_RAE_30001611162862_CDC_22_09_2018_au_21_12_2018.xlsx'
    df = read_file(path,file,sheet_name = 'HOTEL DE VILLE_Pts_10')
    
    #read second file with temperature data
    file = 'donnees_temp_bordeaux_2018.csv'
    df_temp = read_file(path,file,delimiter = ';')
    
    #change the date column from strings to recognised datetime object
    df_temp['Date'] =  pd.to_datetime(df_temp['Date'], format='%d/%m/%Y %H:%M') #format can be customised
    df_temp.sort_values(by=['Date'],inplace = True)                             #dates were not in the correct order
    
    main(df,df_temp,datetime.datetime(2018,1,1,0,0),jours_ferie = ['11-01'], periode_echauffe = {'start':datetime.datetime(2018,11,20,0,0),'end':None})