#machine learning functions used which require xgbosst and sklearn (that two models make problems with pytest)

#libraries neded for it 
import numpy as np
import random as random
import time
import pandas as pd
import scipy
#for confidence intervalls
from scipy.stats import beta
from scipy import stats
#for fitting of x y data 
from scipy.optimize import curve_fit
#ml algorothms
from xgboost import XGBRegressor
from xgboost import XGBClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
#metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#other basic 
from pathlib import Path
import os
import pickle

#returns fractions of wrong  predicted
def perwrong(conf_matrix):
    return 1-(np.sum(conf_matrix)-conf_matrix[0,1]-conf_matrix[1,0])/np.sum(conf_matrix)



#feature_train, target_train, feature_test, target_train, max depth of xgb, needs always be set *6 is equal to default), optional regularization alpha (larger less overfitting)
def do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=0,silent=False):
    start_time=time.time()
    #no regularization option
    if reg==0:
        regxl27=XGBRegressor(max_depth=max_depth).fit(feature_train, target_train)
    else:
        regxl27=XGBRegressor(max_depth=max_depth,reg_alpha=reg).fit(feature_train, target_train)        
    stop_time=time.time()
    print(f"xgb took {round(stop_time-start_time,4)} seconds")
    predli1texl27=regxl27.predict(feature_test)
    predli1trxl27=regxl27.predict(feature_train)
    test_scatter=np.std(predli1texl27-target_test)
    train_scatter=np.std(predli1trxl27-target_train)
    print(f"standard deviation of test {round(np.std(target_test),4)} points")
    print(f"standard deviation of train {round(np.std(target_train),4)}  points")
    print(f"standard deviation of prediction-test {round(test_scatter,4)} points")
    print(f"standard deviation of prediction-train {round(train_scatter,4)} points")
    #copy result to array which can be used by other function
    ar=np.zeros((4))
    ar[0]=reg
    ar[1]=max_depth
    ar[2]=train_scatter
    ar[3]=test_scatter
    return ar


#feature, target(train), #feature, target(test), max_depth, start_reg, factor of increase, number of steps
#name of output file
def loop_reg(feature_train, target_train, feature_test, target_test,max_depth=6,reg_start=0.01,reg_increase=1.414,reg_steps=20,Save=True,file_name=None,regression=True,silent=False):
    #that takes now some time
    resb=np.zeros((4,reg_steps))
    for i in range(reg_steps):
        if silent==False:
            print(f"doing case {i}")
        regularization=reg_start*reg_increase**i
        #regression
        if regression==True:
            ar=do_xgb(feature_train, target_train, feature_test, target_test,max_depth=max_depth,reg=regularization,silent=silent)
        #classification
        else:
            ar=do_xgb_class(feature_train, target_train, feature_test, target_test,max_depth=max_depth,reg=regularization,silent=silent)
        resb[:,i]=ar
    #saved now only at the end seems stable enough now
    if Save==True:
        np.savetxt(file_name, resb) 
    else:
        return resb
        
#2 grid in l2regularziation and max depth
#parameters: feature of train, target of train, feature of test, target of test,
#minimum max deoth, maximum max depth, minimum l2 regularization,
#factor of increase, number of steps, output file name regression=True default
#save=True default, result saved as file otherwise returns the results 
def loop_reg2(feature_train, target_train, feature_test, target_test,max_depth_start=2,max_depth_stop=3,reg_start=0.1,reg_increase=1.414,reg_steps=20,file_name=None,regression=True,save=True,silent=False):
    #creates file to be saved 
    resb=np.zeros((4,reg_steps,int(max_depth_stop-max_depth_start+1)))
    #regularization grid
    for i in range(reg_steps):
        if silent==False:
            print(f"regularization doing case {i}")
        #max depth grid 
        for j in range(resb.shape[2]):
            regularization=reg_start*reg_increase**i
            max_depth=j+max_depth_start
            #regression
            if regression==True:
                ar=do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization,silent=silent)
            #classification
            else:
                ar=do_xgb_class(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization,silent=silent)
            resb[:,i,j]=ar
    if save==True:        
        np.save(file_name, resb) 
    else:        
        return resb  
        
#for xgb calssfier metric, is percentage wrong
#feature_train, target_train, feature_test, target_train, max depth of xgb, needs always be set *6 is equal to default), optional regularization alpha (larger less overfitting)
def do_xgb_class(feature_train, target_train, feature_test, target_test,max_depth=6,reg=0, silent=False):
    start_time=time.time()
    #no regularization option
    if reg==0:
        regxl27=XGBClassifier(max_depth=max_depth).fit(feature_train, target_train)
    else:
        regxl27=XGBClassifier(max_depth=max_depth,reg_alpha=reg).fit(feature_train, target_train)        
    stop_time=time.time()
    if silent==False:
        print(f"xgb took {round(stop_time-start_time,4)} seconds")
    predtest=regxl27.predict(feature_test)
    predtrain=regxl27.predict(feature_train)
    conf_train = confusion_matrix(target_train, predtrain)
    conf_test = confusion_matrix(target_test, predtest)    
    test=perwrong(conf_test)
    train=perwrong(conf_train)
    if silent==False:
        print(f"percentage wrong test {round(test*100,2)}")
        print(f"percentage wrong train {round(train*100,2)} ")
    #copy result to array which can be used by other function
    ar=np.zeros((4))
    ar[0]=reg
    ar[1]=max_depth
    ar[2]=train
    ar[3]=test
    return ar

#find (and run) best xgbosst (regression and classification) of a collection
#parameters are list of the files with the metric and parameters, train_features, train_targtes, whether regression (default) or classification
def find_best(list_inputs,feature_train,target_train,output_file_name,regression=True):
    a=np.loadtxt(list_inputs[0])
    all_metrics=np.zeros((5,len(list_inputs),a.shape[1]))

    for i in range(len(list_inputs)):
        a=np.loadtxt(list_inputs[i])
        all_metrics[0:4,i,:]=a
    #first just using minimum in data
    s1=np.unravel_index(np.argmin(all_metrics[3,:,:]),all_metrics[3,:,:].shape)
    s2=np.argsort(all_metrics[3,:,:],axis=None)
    if regression==True:
        #to the minium seems fine for regression 
        print(f"minimum of {round(all_metrics[3,s1[0],s1[1]],2)} points is at alpha={round(all_metrics[0,s1[0],s1[1]],2)} and max_depth={int(all_metrics[1,s1[0],s1[1]])}")
        reg4=XGBRegressor(max_depth=int(all_metrics[1,s1[0],s1[1]]),reg_alpha=all_metrics[0,s1[0],s1[1]]).fit(feature_train, target_train)
        #and save it
        reg4.save_model(output_file_name)
    else:
        #for classification the minimum seems not good defined
        #but many choices get a similar floor value. The actual minum is likely dependent on the eaxct test sample and thus not necessary reliable. To chose a more relaible, the following procedure is used. Quantile in the test sample are calcauted from 5% onwards in steps of 10%. For that only max depth 5 and larger is used because the minium is always is those for classification. 
        #This quantiles are then used to define the allowed values of the test metric, it is always the one of 15%, it is enlarged, when the quantile slope is getting lower still outside of it. 
        #That define the maximum allowed metric value
        r=np.quantile(all_metrics[3,4:8,:],[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
        print("quantiles of larger max depth half")
        print(r)
        #use them until they get larger again but at least the first 15%
        max_wrong=r[1]
        c=0
        while r[c+2]-r[c+1] <=(r[c+1]-r[0])/(1+c):
            max_wrong=r[c+2]
            c+=1
        print(f"accepted percentage  is {round(100*max_wrong,2)}")  
        #get maximum alpha within this limit, by increasing in alpha in loop start value is mimumum
        #loop goes the other way in max_depth to have this as small as possible
        value=[all_metrics[1,s1[0],s1[1]],all_metrics[0,s1[0],s1[1]]]
        per=all_metrics[3,s1[0],s1[1]]
        for j in range(all_metrics.shape[2]): 
            for i in range(all_metrics.shape[1]):
                if all_metrics[3,7-i,j]<=max_wrong:
                    value[0]=all_metrics[1,7-i,j]
                    value[1]=all_metrics[0,7-i,j] 
                    per=all_metrics[3,7-i,j] 
        print(f"minimum of {round(100*all_metrics[3,s1[0],s1[1]],2)} % is at alpha={round(all_metrics[0,s1[0],s1[1]],2)} and max_depth={int(all_metrics[1,s1[0],s1[1]])}")
        print(f"used of {round(100*per,2)} % is at alpha={round(value[1],2)} and max_depth={int(value[0])}")        
        reg4=XGBClassifier(max_depth=int(value[0]),reg_alpha=value[1]).fit(feature_train, target_train)
        #and save it
        reg4.save_model(output_file_name)  

#needs features, target, number of subunits, seed optional
#gives list of features and targets in these folds
#none means no conditions on content of target
#otherwise number of 1 is made within tolerance just by random trying last subset can deverge more 
def create_k(features,target,k,seed=None,tolerance=None,silent=False,max_its=1000):
    av_val=target.sum()/k
    counter=0
    list_feature=[]
    list_target=[]
    n=0
    for i in range(k-1):
        if tolerance==None:
            features,feature_test,target,target_test=train_test_split(features,target,train_size=1-1/(k-n), shuffle=True, random_state=seed)
        else:
            features_int,feature_test_int,target_int,target_test_int=train_test_split(features,target,train_size=1-1/(k-n), shuffle=True, random_state=seed+counter) 
            counter+=1
            while counter<max_its and np.abs(target_test_int.sum()-av_val)>tolerance:
                features_int,feature_test_int,target_int,target_test_int=train_test_split(features,target,train_size=1-1/(k-n), shuffle=True, random_state=seed+counter) 
                counter+=1
            features,feature_test,target,target_test=features_int,feature_test_int,target_int,target_test_int    
        if i!=k-2:
            list_feature.append(feature_test)
            list_target.append(target_test)
            if silent==False:
                print(f"{i} fold contains {target_test.sum()} class 1")
            n+=1
        else:  
            list_feature.append(feature_test)
            if silent==False:
                print(f"{i} fold contains {target_test.sum()} class 1")                      
            list_target.append(target_test)
            list_feature.append(features)
            list_target.append(target) 
            if silent==False:
                print(f"{i+1} fold contains {target.sum()} class 1")                         
    if silent==False and tolerance!=None:
        print(f"needed {counter} trials")    
    return list_feature,list_target        

#selects test and train from 2 k fold lists
# list_features, list_target, number selected for test
def sel_test_k(list_features,list_target,k,silent=False):
    if k>=len(list_features):
        print("not existing element tried")
    else:
        feature_test=list_features[k]
        target_test=list_target[k]  
        list_features_train=[]
        list_target_train=[]
        for i in range(len(list_features)):
            if i!=k:
                list_features_train.append(list_features[i])
                list_target_train.append(list_target[i])
        feature_train=pd.concat(list_features_train,axis=0, ignore_index=True)
        target_train=pd.concat(list_target_train,axis=0, ignore_index=True)    
        return feature_train, feature_test, target_train, target_test
    
#this loops also over k-fold cross validations
#this also loops over the maximum depth possible
def loop_reg_cross(list_features,list_targets,max_depth_start=2,max_depth_stop=3,reg_start=0.1,reg_increase=1.414,reg_steps=20,file_name=None,regression=True,save=True,silent=False,method="xgboost"):
    if silent==False:
        print(f"doing {len(list_features)} cross validation with {method}")
    #creates intermediate file are K results
    resk=np.zeros((4,reg_steps,int(max_depth_stop-max_depth_start+1),len(list_features)))
    #average of the k
    resb=np.zeros((4,reg_steps,int(max_depth_stop-max_depth_start+1)))
    #regularization grid
    for k in range(len(list_features)):
        feature_train,feature_test,target_train,target_test=sel_test_k(list_features,list_targets,k,silent=silent)
        for i in range(reg_steps):
            if silent==False:
                print(f"regularization doing case {i}")
            #max depth grid 
            for j in range(resb.shape[2]):
                regularization=reg_start*reg_increase**i
                max_depth=j+max_depth_start
                #regression
                if regression==True and method=='xgboost':
                    ar=do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization,silent=silent)
                #classification
                elif method=='xgboost':
                    ar=do_xgb_class(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization,silent=silent)
                resk[:,i,j,k]=ar
    #average of the k
    resb=np.mean(resk,3)    
    if save==True:        
        np.save(file_name, resb) 
    else:        
        return resb  
    
    
#feature, target, k fold, and all parameters from loop_reg_cross
#this here has save2 and file_name 2 that it does not interfere with loop_reg_cross which use without 2 
def cross_fold_fit(features,target,folds,seed=None,max_depth_start=2,max_depth_stop=3,reg_start=0.1,reg_increase=1.414,reg_steps=20,file_name=None,regression=True,save=True,silent=False,method="xgboost",tolerance=None,max_its=1000):
    #create k subsets list
    list_features,list_target=create_k(features,target,folds,seed=seed,silent=silent,tolerance=tolerance,max_its=max_its)
    #fit it
    loop_reg_cross(list_features,list_target,max_depth_start=max_depth_start,max_depth_stop=max_depth_stop,reg_start=reg_start,reg_increase=reg_increase,reg_steps=reg_steps,regression=regression,file_name=file_name,save=save,silent=silent,method=method)
    
#Function for getting the best fit 
def select_best_fit(results):
    #best error improved
    best_err=100
    #best max depth
    best_depth=0
    #check regukarizations
    for i in range(results.shape[2]):
        x_trial63=results[:,:,i]
        print(f"for max depth of {int(results[1,0,i])}")
        print(f"best regularization is {x_trial63[0,np.argmin(100*x_trial63[3])]} where in test {np.round(np.min(100*x_trial63[3]),2)} % are wrong ")
        #check whether imrpoved
        if 100*x_trial63[3,np.argmin(100*x_trial63[3])]<best_err:
            x_trial62=x_trial63
            best_depth=int(results[1,0,i])
            best_err=100*x_trial63[3,np.argmin(100*x_trial63[3])]
            best_reg=x_trial63[0,np.argmin(100*x_trial63[3])] 
    #return array of best max deoth and best regularization and max_depth
    return x_trial62, best_reg, best_depth    

#applies several fold fits on the same data 
#feature_list, target_list, other_feature, is the data on which it is applied, parameters of fit to be used 
def loop_fold(list_features,list_targets,other_feature,max_depth,reg,regression=False,silent=False,method="xgboost"):
    results=np.zeros((list_targets[0].nunique(),other_feature.shape[0],len(list_features)))
    for k in range(len(list_features)):
        feature_train,feature_test,target_train,target_test=sel_test_k(list_features,list_targets,k,silent=silent)
        if method=="xgboost":
            if silent==False:
                print(f"xgboost fit of fold {k}")
            xmodel=XGBClassifier(max_depth=max_depth,reg_alpha=reg).fit(feature_train, target_train)
            results[:,:,k]=xmodel.predict_proba(other_feature).T
    res=np.mean(results,2)  
    return res

    
#applies several fold fits on the same data predcit on out of fold 
#feature_list, target_list, other_feature, is the data on which it is applied, parameters of fit to be used 
def loop_fold_out(list_features,list_targets,max_depth,reg,regression=False,silent=False,method="xgboost"):
    c=0
    for k in range(len(list_features)):
        feature_train,feature_test,target_train,target_test=sel_test_k(list_features,list_targets,k,silent=silent)
        c+=feature_test.shape[0]
    results=np.zeros((list_targets[0].nunique()+1+list_features[0].shape[1],c))
    c=0
    for k in range(len(list_features)):
        feature_train,feature_test,target_train,target_test=sel_test_k(list_features,list_targets,k,silent=silent)
        if method=="xgboost":
            if silent==False:
                print(f"xgboost fit of fold {k}")
            xmodel=XGBClassifier(max_depth=max_depth,reg_alpha=reg).fit(feature_train, target_train)
            results[0:feature_test.shape[1],c:c+feature_test.shape[0]]=feature_test.T
            results[feature_test.shape[1]:feature_test.shape[1]+1,c:c+feature_test.shape[0]]=target_test
            results[feature_test.shape[1]+1:results.shape[0],c:c+feature_test.shape[0]]=xmodel.predict_proba(feature_test).T 
            c+=feature_test.shape[0]
    return results


#feature, target, k fold, performance out of fold 
#only one depth and regularziation is here used 
def cross_fold_apply_itself(features,target,folds,seed=None,max_depth=2,reg=0.1,file_name=None,regression=False,save=False,silent=False,method="xgboost",tolerance=None,max_its=1000):
    #create k subsets list
    list_features,list_targets=create_k(features,target,folds,seed=seed,silent=silent,tolerance=tolerance,max_its=max_its)
    #fit it
    res=loop_fold_out(list_features,list_targets,max_depth=max_depth,reg=reg,regression=regression,silent=silent,method=method)
    if save==True:        
        np.save(file_name, res) 
    else:   
        return res

#feature, target, k fold, other_feature is the data on which it is applied
#only one depth and regularziation is here used 
def cross_fold_apply(features,target,folds,other_feature,seed=None,max_depth=2,reg=0.1,file_name=None,regression=False,save=False,silent=False,method="xgboost",tolerance=None,max_its=1000):
    #create k subsets list
    list_features,list_targets=create_k(features,target,folds,seed=seed,silent=silent,tolerance=tolerance,max_its=max_its)
    #fit it
    res=loop_fold(list_features,list_targets,other_feature=other_feature,max_depth=max_depth,reg=reg,regression=regression,silent=silent,method=method)
    if save==True:        
        np.save(file_name, res) 
    else:   
        return res
    
