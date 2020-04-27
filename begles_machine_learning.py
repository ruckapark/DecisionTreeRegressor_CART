# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:58:01 2019

@author: gruck
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def main(directory,input_training,input_prediction,levels,output):
    #%% read in input file
    os.chdir(directory)
    df = pd.read_csv(input_training,delimiter = ';',index_col = 0)
    df_pred = pd.read_csv(input_prediction,delimiter = ';',index_col = 0)
    df_pred.dropna(inplace = True)
    
    #%% arbre decisionel
    
    from sklearn.tree import DecisionTreeRegressor #decision tree model
    from sklearn.ensemble import AdaBoostRegressor #for better fit of predictions
    
    X = df[['Temperature','weekday','echauffement','hm']]   #vars expliqs
    Y = df['PS ATTEINTE']                                   #var à expliquer
    
    tree = AdaBoostRegressor(DecisionTreeRegressor(max_depth=levels), n_estimators=20)  #boosting de l'arbre
    tree.fit(X,Y)
    
    #export_graphviz(tree, out_file = 'tree.txt')   #si on veut visualiser l'arbre
    
    #%%
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33)  #training set #0.67 of training data
    
    #test at various tree depths to capture variance explained
    scores = []
    for i in range(1,20):
        tree1 = DecisionTreeRegressor(max_depth=i)
        tree1.fit(X_train,Y_train)
        
        valid_pred = tree1.predict(X_test)          #output data in each case
        scores.append(tree1.score(X_test,Y_test))   #use R^2 scores as an idea of how many tree levels should be used
    
    #%%final prediction
    
    train_pred = tree.predict(X)
    test_pred = tree.predict(df_pred[['Temperature','weekday','echauffement','hm']])
    #for underestimate multiply by empiric factor
    
    df_pred['PS ATTEINTE'] = test_pred
    data2018 = pd.DataFrame(pd.concat([df_pred,df]))
    data2018.sort_index(inplace = True)
    
    data2018.to_csv(output,sep = ';')
    plt.figure()
    plt.plot(scores)
    plt.title('R squared explained for n levels in regression tree')
    plt.figure()
    data2018['PS ATTEINTE'].plot()
    plt.title('Courbe de Charge')
    
    
##################################################
#%% MAIN CODE
##################################################

if __name__ == "__main__":
    directory = r'C:\Users\gruck\Documents\Begles'
    input_training = 'begles_3month_Data.csv'
    input_prediction = 'begles_prediction_Data.csv'
    levels = 15
    output = 'data2018.csv'
    
    main(directory,input_training,input_prediction,levels,output)