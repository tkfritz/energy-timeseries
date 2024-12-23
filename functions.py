#machine learning functions used which require xgbosst and sklearn (that two models make problems with pytest)

#libraries neded for it 
import numpy as np
import random as random
from datetime import date, time, datetime, timedelta
import pandas as pd
import scipy as sp
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
#matplotlib
from matplotlib import pyplot as plt
#easter date
from dateutil.easter import *

#returns fractions of wrong  predicted
def perwrong(conf_matrix):
    return 1-(np.sum(conf_matrix)-conf_matrix[0,1]-conf_matrix[1,0])/np.sum(conf_matrix)



#feature_train, target_train, feature_test, target_train, max depth of xgb, needs always be set *6 is equal to default), optional regularization alpha (larger less overfitting)
def do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=0,silent=False):
    start_time=datetime.now()
    #no regularization option
    if reg==0:
        regxl27=XGBRegressor(max_depth=max_depth).fit(feature_train, target_train)
    else:
        regxl27=XGBRegressor(max_depth=max_depth,reg_alpha=reg).fit(feature_train, target_train)        
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
    stop_time=datetime.now()
    print(f"xgb took {(stop_time-start_time)} seconds")
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
    
#parameters are: 
#data_frame, number of data points used for prediction, number of data points ignored between data and target, nan excluding
def series_to_supervised(data, n_in=1, offset=0, dropnan=True):
    #create empty data frame and list
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t+offset)
    for i in range(offset, 1+offset):
        cols.append(df.shift(-i))
    # connecting all 
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

#parameters, data, min time pred period in hours/4-1/4, max-min time pred peiod in hours/4-1/4, predictining over how many data_points
#steps in test, model shortcut used, name of file for output
def predict_intervall(df,a,b,c,d,model,output):
    resf=np.zeros((4,b))
    #valid_modelchecks whether model is called
    valid_model=False
    if model=="xgb" or model=="xgboost":
        xmodel3=XGBRegressor()
        print("running XGBoost")
        valid_model=True
    if model=="lin" or model=='linear':
        xmodel3=LinearRegression()
        print("running Linear Regression")
        valid_model=True
    if valid_model==True:
        for i in range(int(a), int(b+a)):
            print(f"running prediction over {(i+1)*15} minutes")
            #convert to supervised data format
            res3=series_to_supervised(df.total_power,c,i)
            res3=pd.DataFrame(res3)
            #split in test and train
            dataa3_train = res3.iloc[:-d,:]
            dataa3_test  = res3.iloc[-d:,:]
            x3_train=dataa3_train.iloc[:,:c]
            x3_test=dataa3_test.iloc[:,:c]
            y3_train=dataa3_train.iloc[:,c]
            y3_test=dataa3_test.iloc[:,c]
            #time scale of prediction
            resf[0,i-int(a)]=15*(i+1)
            #fit and predict
            xmodel3.fit(x3_train,y3_train)
            pred3=xmodel3.predict(x3_test)
            #standard deviation of prediction
            resf[2,i-int(a)]=np.std(pred3-y3_test)
            #compare with shift
            diff6=y3_test.shift(periods=i+1)           
            resf[1,i-int(a)]=np.std(diff6-y3_test)
            #mean value to get percentage
            resf[3,i-int(a)]=np.mean(y3_test)
    else:
        print("no implemented model called")
    np.savetxt(output,resf)
    
#quadrtyic logarithm
def log_quat(x,a,b,c):
    return a+b*np.log10(x)+c*np.log10(x)**2    

def dev_quat(a,b,c):
    return -b/(2*c)
 
def find_best(dat,delta=3):
    min_p=np.argmin(dat[3])
    est=np.zeros((3))
    print(dat[0,min_p])
    print(dat[0:4,min_p-delta:min_p+delta])
    if min_p-delta>=0:
        val,cov=sp.optimize.curve_fit(log_quat,dat[0,min_p-delta:min_p+delta],dat[3,min_p-delta:min_p+delta],p0=est)
    else:
        val,cov=sp.optimize.curve_fit(log_quat,dat[0,0:min_p+delta],dat[3,0:min_p+delta],p0=est)
    print(val)
    min_p1=10**dev_quat(val[0],val[1],val[2])
    return min_p1
    
#parameters data frame, gap to wanted prediction
def find_fit_best_reg(df,gap,now_points=1,test_frac=0.8,max_depth=4,reg_start=0.001,reg_increase=1.414,reg_steps=4,delta=4,filename=None,save=True):
    ser=series_to_supervised(df.total_power,now_points,gap)
    df_ser=pd.DataFrame(ser,columns=["now","to_predict"])
    df_ser.loc[:,'frac_day']=df.loc[0:df_ser.shape[0],'frac_day']
    df_ser.loc[:,'frac_week']=df.loc[0:df_ser.shape[0],'frac_week']
    df_ser.loc[:,'frac_year']=df.loc[0:df_ser.shape[0],'frac_year']
    print(df_ser.shape)
    print(df_ser.columns)
    frac1=int(df_ser.shape[0]*0.8)
    print(frac1)
    ser_train=df_ser.iloc[:frac1,:]
    ser_test=df_ser.iloc[frac1:,:]
    print(reg_steps)
    stat_reg=loop_reg(ser_train.loc[:,['now','frac_day', 'frac_week', 'frac_year']],ser_train.to_predict,ser_test.loc[:,['now','frac_day', 'frac_week', 'frac_year']],ser_test.to_predict,max_depth=max_depth,reg_start=reg_start,reg_increase=reg_increase,reg_steps=reg_steps,Save=False,regression=True,silent=False)
    print(stat_reg)
    best_reg=find_best(stat_reg,delta=delta)
    xmodel=XGBRegressor(alpha=best_reg).fit(df_ser.loc[:,['now','frac_day', 'frac_week', 'frac_year']],df_ser.loc[:,'to_predict'])
    if save==True:
        xmodel.save_model(filename)
    else:
        return xmodel

def fit_many_gaps(df,gap_start=1,gap_steps=2,now_points=1,test_frac=0.8,max_depth=4,reg_start=0.001,reg_increase=1.414,reg_steps=4,delta=4,filename=None,save=True):
    for i in range(gap_steps):
        gap=gap_start+i
        print(gap)
        if gap<10:
            filename="xgb_model_00"+str(gap)+".json"
        elif gap<100:
            filename="xgb_model_0"+str(gap)+".json"
        elif gap<1000:
            filename="xgb_model_"+str(gap)+".json"            
        find_fit_best_reg(df,gap,now_points=now_points,test_frac=test_frac,max_depth=max_depth,reg_start=reg_start,reg_increase=reg_increase,reg_steps=reg_steps,delta=delta,filename=filename,save=True)
       

def transform_projected(df):
    dic2b={'Datum':'Date','Anfang':'Time','Gesamt (Netzlast) [MWh] Originalauflösungen':'total_power_pred','Residuallast [MWh] Originalauflösungen':'residual_power_pred'}
    df.rename(columns=dic2b,inplace=True)
    #drop columns  not there anymore
    #df.drop(['Ende'], axis=1, inplace=True)
    #cpnvert german float to english 
    df['residual_power_pred'] = df['residual_power_pred'].str.replace('.','')
    df['residual_power_pred'] = df['residual_power_pred'].str.replace(',','.').astype(float)/1000.
    df['total_power_pred'] = df['total_power_pred'].str.replace('.','')
    df['total_power_pred'] = df['total_power_pred'].str.replace(',','.').astype(float)/1000.
    #somehow below is not anymore there
    #df['pump_storage_pred'] = df['pump_storage_pred'].str.replace('.','')
    #df['pump_storage_pred'] = df['pump_storage_pred'].str.replace(',','.').astype(float)/1000.
    df['date_time']=pd.to_datetime(df['Date'] + '.' + df['Time'], format='%d.%m.%Y.%H:%M')
    return df


def prepare_input(df,pump=False,end=False,bad_cut=0.9,zero_time=(2015,1,1,0,0),old=False,str_convert=True,version='y_fraction1',newest=False):
    #zero time of model can change later
    zero=datetime(zero_time[0],zero_time[1],zero_time[2],zero_time[3],zero_time[4])
    #new column names
    if old==False and newest==False:
        dic2={'Datum':'Date','Anfang':'Time','Gesamt (Netzlast) [MWh] Originalauflösungen':'total_power','Residuallast [MWh] Originalauflösungen':'residual_power','Pumpspeicher [MWh] Originalauflösungen':'pump_storage'}
        df.rename(columns=dic2,inplace=True)
    if old==False and newest==True:
        dic3={'VD-DM-Alles.tabellenKopf.columnHeader.date_start':'date_time','VD-DM-Alles.tabellenKopf.columnHeader.date_end':'date_time_end','Gesamt (Netzlast) [MWh] Originalauflösungen':'total_power','Residuallast [MWh] Originalauflösungen':'residual_power','Pumpspeicher [MWh] Originalauflösungen':'pump_storage'}
        df.rename(columns=dic3,inplace=True)        
    elif old==True:
        dic1={'Datum':'Date','Uhrzeit':'Time','Gesamt (Netzlast)[MWh]':'total_power','Residuallast[MWh]':'residual_power','Pumpspeicher[MWh]':'pump_storage'}
        df.rename(columns=dic1,inplace=True)
    #drop columns
    if end==True:
        df.drop(['Ende'], axis=1, inplace=True)
    #cpnvert german float to english 
    if str_convert==True:
        df['residual_power'] = df['residual_power'].str.replace('.','')
        df['residual_power'] = df['residual_power'].str.replace('-','0')
        df['residual_power'] = df['residual_power'].str.replace(',','.').astype(float)/1000.
        df['total_power'] = df['total_power'].str.replace('.','')
        df['total_power'] = df['total_power'].str.replace('-','0')    
        df['total_power'] = df['total_power'].str.replace(',','.').astype(float)/1000.
        if pump==True:
            df['pump_storage'] = df['pump_storage'].str.replace('.','')
            df['pump_storage'] = df['pump_storage'].str.replace(',','.').astype(float)/1000.       
    #old not combined date and time         
    if newest==False:        
        df['date_time']=pd.to_datetime(df['Date'] + '.' + df['Time'], format='%d.%m.%Y.%H:%M')
    #new is combined     
    else:
        df['date_time']=pd.to_datetime(df['date_time'], format='%d.%m.%Y %H:%M')
    delta=str(df.loc[0,'date_time']-zero)
    #deltam=time.strftime(delta,'%M')
    days=delta.split(' days ')
    hour=days[1].split(':')
    #difference in fraction of days
    diff_frac=float(days[0])+float(hour[0])/24+float(hour[1])/24/60
    time1=np.zeros((df.shape[0],5))
    for i in range(df.shape[0]):
        time1[i,0]=diff_frac+i/4/24
        time1[i,1]=time1[i,0]%1
        time1[i,2]=(time1[i,0]%7)/7
        time1[i,3]=(time1[i,0]%365.25)/365.25
        time1[i,4]=i/4/24/365.25      
    df['frac_day']=time1[:,1]
    df['frac_week']=time1[:,2]
    df['frac_year']=time1[:,3]
    #works at least for full days, later more checks 
    #helper parameter used below
    df['year']=df.date_time.dt.year.astype(int)
    df['month']=df.date_time.dt.month.astype(int)
    #days from easter of this year
    df['delta_easter']=0
    #days from first march of the year, gest holidayts right given leap years
    df['delta_march']=0
    #works at least for full days, later more checks 
    for i in range(df.shape[0]):
        df['delta_easter'].iloc[i]=(df['date_time'].iloc[i]-pd.to_datetime(easter(df.year.iloc[i]))).days
        if df.month.iloc[i]<2.5:
            df['delta_march'].iloc[i]=(df['date_time'].iloc[i]-datetime(df['year'].iloc[i]-1,3,1)).days
        else:
            df['delta_march'].iloc[i]=(df['date_time'].iloc[i]-datetime(df['year'].iloc[i],3,1)).days            
    #exclude what is zero at the end
    c=0
    while df.loc[df.shape[0]+c-1,'total_power']==0:
        c-=1
    #interpolate linear bad value (only exatcly 0 for now) which are not at the beginning or end 
    for i in range(1,df.shape[0]+c-2):
        if df.loc[i,'total_power']==0:
            df.loc[i,'total_power']=(df.loc[i-1,'total_power']+df.loc[i+1,'total_power'])/2    
    #return of the the needed columns in the right order, and dleta when last is likely bad
    delta=0                   
    if df.loc[df.shape[0]+c-1,'total_power']/df.loc[df.shape[0]+c-2,'total_power']<bad_cut:
        delta=-1
    #return needed output, order matters    
    if version=='y_fraction1':    
        return df.loc[:df.shape[0]+c-1+delta,['total_power','frac_day', 'frac_week', 'frac_year','date_time']]
    elif version=='d_easter1':    
        return df.loc[:df.shape[0]+c-1+delta,['frac_day','frac_week','delta_easter','total_power','date_time']] 
    elif version=='d_march1':    
        return df.loc[:df.shape[0]+c-1+delta,['frac_day','frac_week','delta_march','total_power','date_time']]    
 
    
#plotting function 
def plot_prediction(power_newest,prediction_newest,plot_error=False):
    plt.plot(power_newest['date_time'],(power_newest['total_power']*4),'-',ms=1,color='blue',label='observed')   
    plt.plot(prediction_newest.date_time,prediction_newest.consumption_cleaned,color='red',label='prediction')
    if plot_error==True:
        plt.plot(prediction_newest.date_time,prediction_newest.consumption_cleaned+prediction_newest.error,color='orange',label='1 sigma band')
        plt.plot(prediction_newest.date_time,prediction_newest.consumption_cleaned-prediction_newest.error,color='orange')
    plt.xlabel("date")
    plt.ylabel("consumption [GW]") 
    plt.legend(loc="best")
    plt.title("current prediction")
    max_t=prediction_newest.loc[prediction_newest.shape[0]-1,'date_time']+timedelta(days=1)
    min_t=prediction_newest.loc[prediction_newest.shape[0]-1,'date_time']+timedelta(days=-2)
    year_stop=max_t.year
    month_stop=max_t.month
    day_stop=max_t.day
    year_start=min_t.year
    month_start=min_t.month
    day_start=min_t.day
    plot_start = datetime(year_start, month_start, day_start)
    if plot_error==False:
        min_pred=prediction_newest.consumption_cleaned.min()
        max_pred=prediction_newest.consumption_cleaned.max()
    else:
        min_pred=(prediction_newest.consumption_cleaned-prediction_newest.error).min()
        max_pred=(prediction_newest.consumption_cleaned+prediction_newest.error).max()    
    power_sel=power_newest[(power_newest.date_time==plot_start)]
    min_actual=4*power_newest[power_sel.index[0]:].total_power.min()
    max_actual=4*power_newest[power_sel.index[0]:].total_power.max()
    min_power=min(min_pred,min_actual)
    max_power=max(max_pred,max_actual)
    plt.xlim(datetime(year_start,month_start,day_start),datetime(year_stop,month_stop,day_stop))
    plt.ylim(min_power*0.99,max_power*1.01)    
    
    
def find_data(end_x='json',data='Realisierter_Stromverbrauch_',myPath='/home/tobias/ml-testing/energy/energy-timeseries',version='y_fraction1'):
    if version=='y_fraction1':
        start_x='xgb_model_'
    if version=='d_easter1':
        start_x='xgb2e_model_'
        errors=np.loadtxt("xgb2e_model_error_offset.txt")
    if version=='d_march1':
        start_x='xgb2m_model_'
        errors=np.loadtxt("xgb2m_model_error_offset.txt")        
    models=[f for f in os.listdir(myPath) 
        if (f.startswith(start_x)) and  (f.endswith(end_x) )] 
    models.sort()
    data=[f for f in os.listdir(myPath) 
        if (f.startswith(data))]
    data.sort()
    if version=='y_fraction1': 
        return models, data
    if version=='d_easter1': 
        return models, data, errors      
    if version=='d_march1': 
        return models, data, errors     
    
#parameters, most recent features, list of model,delta ts, standard is just every 0.25 h from models, delta, means whether start_time is reduced sometimes also point is bad, like in  Realisierter_Stromverbrauch_202402260600_202403080559_Viertelstunde.csv (residual and pump are missing then could be a sign), now cleaned in input when 10% off from second alst data point, now linear interpolation of point when 4% of for 2 cloest and 2 second closets enighbor is heuristic, also saved now, better (justified) interpolation done at some point, both now saved
def predict_from_now(data,models,errors=None,deltas=None,silent=False,delta=0,version='y_fraction1'):
    if silent==False:    
        print(data)
    if deltas==None:
        deltas=np.zeros((len(models)))
        for i in range(len(models)):
            deltas[i]=0.25+i/4
    #3 error to be added at some point        
    res=np.zeros((3,len(models)))  
    res[0,:]=deltas
    for i in range(len(models)):
        if silent==False:
            print(f"prediction of model {i} dones")
        xmodel=XGBRegressor()
        xmodel.load_model(models[i])
        #predict needs more than 1 data point to work 
        #oldest model no error and nom offset
        if version=='y_fraction1':
            res[1,i]=xmodel.predict(data.iloc[:,0:4])[-1+delta]*4
        #other have     
        elif version=='d_easter1':
            #predict
            pred=xmodel.predict(data.iloc[:,0:4])[-1+delta]
            #use absolute adjustments
            res[1,i]=(pred+errors[2,i])*4   
            res[2,i]=errors[1,i]*4
        elif version=='d_march1':
            pred=xmodel.predict(data.iloc[:,0:4])[-1+delta]
            res[1,i]=(pred+errors[2,i])*4   
            res[2,i]=errors[1,i]*4
    #make data frame 
    df=pd.DataFrame(res.T,columns=['hours','consumption','error'])
    for i in range(df.shape[0]):
        df.loc[i,'date_time']=data.iloc[data.shape[0]-1+delta,data.shape[1]-1]+timedelta(hours=df['hours'][i])
    #also write it that the predictions can be investiagted at some point
    year=data.iloc[data.shape[0]-1+delta,data.shape[1]-1].year
    month=data.iloc[data.shape[0]-1+delta,data.shape[1]-1].month    
    day=data.iloc[data.shape[0]-1+delta,data.shape[1]-1].day   
    hour=data.iloc[data.shape[0]-1+delta,data.shape[1]-1].hour    
    minute=data.iloc[data.shape[0]-1+delta,data.shape[1]-1].minute                   
    return df,year,month,day, hour, minute         


def quadratic(x,a,b,c):
    return a+b*x+c*x**2


def clean_prediction(df,year,month,day, hour, minute,silent=True,mode='linear1',version='y_fraction1',clean_lim=0.04):
    #create now new cleaned column 
    #df['consumption_cleaned']=df['consumption']
    if mode=='linear1':
        for i in range(df.shape[0]):
            if i>=2 and i<df.shape[0]-2:            
               #only correct when larger x for neighbor interpolation of 2 clest and 2 second closest neighbors
               if np.abs(df.loc[i,'consumption']-df.loc[i-2,'consumption']/2-df.loc[i+2,'consumption']/2)/(df.loc[i-2,'consumption']/2+df.loc[i+2,'consumption']/2)>clean_lim  and np.abs(df.loc[i,'consumption']-df.loc[i-1,'consumption']/2-df.loc[i+1,'consumption']/2)/(df.loc[i-1,'consumption']/2+df.loc[i+1,'consumption']/2)>clean_lim:
                    df.loc[i,'consumption_cleaned']=(df.loc[i-1,'consumption']+df.loc[i+1,'consumption'])/2  
                    if silent==False:
                        print(f"{i} is cleaned")
               else:   
                    df.loc[i,'consumption_cleaned']=df.loc[i,'consumption']
            else:   
                df.loc[i,'consumption_cleaned']=df.loc[i,'consumption']        
    elif mode=='quadratic1':
        for i in range(df.shape[0]):
            if i>=2 and i<df.shape[0]-2:
                err=np.zeros((4,5))
                err[0]=[-2.,-1.,0.,1.,2.]
                err[1]=df.loc[i-2:i+2,'consumption']
                if version=='y_fraction1':
                    #set one set the scipy below workds
                    err[2]=1.
                    err[3]=1.
                else:    
                    err[2]=df.loc[i-2:i+2,'error']
                    err[3]=df.loc[i-2:i+2,'error']
                x2=0
                for j in range(5):
                    x=i-2+j-2
                    y=i+2+j+2
                    if x>=0 and y<df.shape[0]:
                        if np.abs(df.loc[i+j-2,'consumption']-df.loc[i-2+j-2,'consumption']/2-df.loc[i+2+j-2,'consumption']/2)/(df.loc[i-2+j-2,'consumption']/2+df.loc[i+2+j-2,'consumption']/2)>clean_lim  and np.abs(df.loc[i+j-2,'consumption']-df.loc[i-1+j-2,'consumption']/2-df.loc[i+1+j-2,'consumption']/2)/(df.loc[i-1+j-2,'consumption']/2+df.loc[i+1+j-2,'consumption']/2)>clean_lim:
                            #error is increased by factor 10, means in practice irrelevant easier then remove it
                            err[3,j]=err[2,j]*10
                            x2=2
                            if silent==False:
                                print(f"{i} is error  increased")
                                print(err)
                est=np.zeros((3))
                val,cov=sp.optimize.curve_fit(quadratic,err[0],err[1],sigma=err[3],absolute_sigma=True,p0=est)
                df.loc[i,'consumption_cleaned']=val[0]
                if silent==False and x2==2:
                    print(df.loc[i,'consumption_cleaned'])
            else:
                df.loc[i,'consumption_cleaned']=df.loc[i,'consumption']
                
    df.to_csv(version+'_prediction_'+str(year)+'_'+str(month)+'_'+str(day)+'_'+str(hour)+'_'+str(minute)+'.csv',sep=',') 
    return df


#parameter, version of models, whetehr process is printed, whether error is used in Figure
def pipeline(version='y_fraction1',silent=True,plot_error=False,mode='linear1',new_level=0):
    #get model and data file lists
    if version=='y_fraction1':
        models,data=find_data(version=version)
        #create dummy errors to simplyfy prediction below
        errors=np.zeros((3,len(models)))
        #create non zero error to avoid error problem
        errors[1]=1.
    elif version=='d_easter1':
        models,data,errors=find_data(version=version)   
    elif version=='d_march1':
        models,data,errors=find_data(version=version)           
    #also error/offset file if exist    
    #last in list is newest 
    new_real=pd.read_csv(data[-1],delimiter=';')
    #prepocessing including some data cleaning
    power_newest=prepare_input2(new_real,new_level=new_level,version=version,str_convert=True)
    #apply prediction
    prediction_newest,y,mo,d,h,mi=predict_from_now(power_newest.loc[power_newest.shape[0]-3:power_newest.shape[0],:],models[:],errors,silent=silent,version=version)
    #clean prediction
    prediction_newest=clean_prediction(prediction_newest,y,mo,d,h,mi,silent=silent,mode=mode,version=version)
    #plot prediction
    plot_prediction(power_newest.iloc[:,:],prediction_newest,plot_error=plot_error)
    
    

#now hidden parameters for d_easter1 model 
#parameter, version of models, whetehr process is printed, whether error is used in Figure
#does not work currently because if changes in version of data 
def pipeline_v2(version='d_easter1',silent=True,plot_error=False,mode='quadratic1',newest=True,old=False):
    #get model and data file lists
    if version=='y_fraction1':
        models,data=find_data(version=version)
        #create dummy errors to simplyfy prediction below
        errors=np.zeros((3,len(models)))
        #create non zero error to avoid error problem
        errors[1]=1.
    elif version=='d_easter1':
        models,data,errors=find_data(version=version)  
    elif version=='d_march1':
        models,data,errors=find_data(version=version)          
    #also error/offset file if exist    
    #last in list is newest 
    new_real=pd.read_csv(data[-1],delimiter=';')
    power_newest=prepare_input(new_real,newest=newest,old=old,version=version,str_convert=True)
    #apply prediction
    prediction_newest,y,mo,d,h,mi=predict_from_now(power_newest.loc[power_newest.shape[0]-3:power_newest.shape[0],:],models[:],errors,silent=silent,version=version)
    #clean prediction
    prediction_newest=clean_prediction(prediction_newest,y,mo,d,h,mi,silent=silent,mode=mode,version=version)
    #plot prediction
    plot_prediction(power_newest.iloc[:,:],prediction_newest,plot_error=plot_error)
    
#this function sets standard parameter to march model that function can be called without parameters
#parameter, version of models, whetehr process is printed, whether error is used in Figure
def pipeline_v3(version='d_march1',silent=True,plot_error=False,mode='quadratic1',new_level=2):
    #get model and data file lists
    if version=='y_fraction1':
        models,data=find_data(version=version)
        #create dummy errors to simplyfy prediction below
        errors=np.zeros((3,len(models)))
        #create non zero error to avoid error problem
        errors[1]=1.
    elif version=='d_easter1':
        models,data,errors=find_data(version=version)   
    elif version=='d_march1':
        models,data,errors=find_data(version=version)         
    #also error/offset file if exist    
    #last in list is newest 
    new_real=pd.read_csv(data[-1],delimiter=';')
    power_newest=prepare_input2(new_real,new_level=new_level,version=version,str_convert=True)
    #apply prediction
    prediction_newest,y,mo,d,h,mi=predict_from_now(power_newest.loc[power_newest.shape[0]-3:power_newest.shape[0],:],models[:],errors,silent=silent,version=version)
    #clean prediction
    prediction_newest=clean_prediction(prediction_newest,y,mo,d,h,mi,silent=silent,mode=mode,version=version)
    #plot prediction
    plot_prediction(power_newest.iloc[:,:],prediction_newest,plot_error=plot_error)
 
#now hidden parameters for d_easter1 model 
#parameter, version of models, whetehr process is printed, whether error is used in Figure
def pipeline_v3b(version='d_easter1',silent=True,plot_error=False,mode='quadratic1',new_level=2):
    #get model and data file lists
    if version=='y_fraction1':
        models,data=find_data(version=version)
        #create dummy errors to simplyfy prediction below
        errors=np.zeros((3,len(models)))
        #create non zero error to avoid error problem
        errors[1]=1.
    elif version=='d_easter1':
        models,data,errors=find_data(version=version)   
    elif version=='d_march1':
        models,data,errors=find_data(version=version)         
    #also error/offset file if exist    
    #last in list is newest 
    new_real=pd.read_csv(data[-1],delimiter=';')
    power_newest=prepare_input2(new_real,new_level=new_level,version=version,str_convert=True)
    #apply prediction
    prediction_newest,y,mo,d,h,mi=predict_from_now(power_newest.loc[power_newest.shape[0]-3:power_newest.shape[0],:],models[:],errors,silent=silent,version=version)
    #clean prediction
    prediction_newest=clean_prediction(prediction_newest,y,mo,d,h,mi,silent=silent,mode=mode,version=version)
    #plot prediction
    plot_prediction(power_newest.iloc[:,:],prediction_newest,plot_error=plot_error)
    
#now hidden parameters for y_fraction1 model 
#parameter, version of models, whetehr process is printed, whether error is used in Figure
def pipeline_v3c(version='y_fraction1',silent=True,plot_error=False,mode='quadratic1',new_level=2):
    #get model and data file lists
    if version=='y_fraction1':
        models,data=find_data(version=version)
        #create dummy errors to simplyfy prediction below
        errors=np.zeros((3,len(models)))
        #create non zero error to avoid error problem
        errors[1]=1.
    elif version=='d_easter1':
        models,data,errors=find_data(version=version)   
    elif version=='d_march1':
        models,data,errors=find_data(version=version)         
    #also error/offset file if exist    
    #last in list is newest 
    new_real=pd.read_csv(data[-1],delimiter=';')
    power_newest=prepare_input2(new_real,new_level=new_level,version=version,str_convert=True)
    #apply prediction
    prediction_newest,y,mo,d,h,mi=predict_from_now(power_newest.loc[power_newest.shape[0]-3:power_newest.shape[0],:],models[:],errors,silent=silent,version=version)
    #clean prediction
    prediction_newest=clean_prediction(prediction_newest,y,mo,d,h,mi,silent=silent,mode=mode,version=version)
    #plot prediction
    plot_prediction(power_newest.iloc[:,:],prediction_newest,plot_error=plot_error)
    
#now hidden parameters for y_fraction1 model 
#parameter, version of models, whetehr process is printed, whether error is used in Figure
#does not work currently because if changes in version of data 
def pipeline_v1(version='y_fraction1',silent=True,plot_error=False,mode='quadratic1'):
    #get model and data file lists
    if version=='y_fraction1':
        models,data=find_data(version=version)
        #create dummy errors to simplyfy prediction below
        errors=np.zeros((3,len(models)))
        #create non zero error to avoid error problem
        errors[1]=1.
    elif version=='d_easter1':
        models,data,errors=find_data(version=version)   
    elif version=='d_march1':
        models,data,errors=find_data(version=version)          
    #also error/offset file if exist    
    #last in list is newest 
    new_real=pd.read_csv(data[-1],delimiter=';')
    #prepocess including some data cleaning
    power_newest=prepare_input(new_real,version=version)
    #apply prediction
    prediction_newest,y,mo,d,h,mi=predict_from_now(power_newest.loc[power_newest.shape[0]-3:power_newest.shape[0],:],models[:],errors,silent=silent,version=version)
    #clean prediction
    prediction_newest=clean_prediction(prediction_newest,y,mo,d,h,mi,silent=silent,mode=mode,version=version)
    #plot prediction
    plot_prediction(power_newest.iloc[:,:],prediction_newest,plot_error=plot_error)    

#new version easier to update column nature 
def prepare_input2(df,pump=False,end=False,bad_cut=0.9,zero_time=(2015,1,1,0,0),str_convert=True,version='y_fraction1',new_level=0):
    #zero time of model can change later
    zero=datetime(zero_time[0],zero_time[1],zero_time[2],zero_time[3],zero_time[4])
    if new_level==0:
        dic1={'Datum':'Date','Uhrzeit':'Time','Gesamt (Netzlast)[MWh]':'total_power','Residuallast[MWh]':'residual_power','Pumpspeicher[MWh]':'pump_storage'}
        df.rename(columns=dic1,inplace=True)
    #new column names
    elif new_level==1:
        dic2={'Datum':'Date','Anfang':'Time','Gesamt (Netzlast) [MWh] Originalauflösungen':'total_power','Residuallast [MWh] Originalauflösungen':'residual_power','Pumpspeicher [MWh] Originalauflösungen':'pump_storage'}
        df.rename(columns=dic2,inplace=True)   
    elif new_level==2:
        dic3={'Datum von':'date_time','Datum bis':'date_time_end','Gesamt (Netzlast) [MWh] Originalauflösungen':'total_power','Residuallast [MWh] Originalauflösungen':'residual_power','Pumpspeicher [MWh] Originalauflösungen':'pump_storage'}
        df.rename(columns=dic3,inplace=True)     
    #drop columns
    if end==True and new_level<2:
        df.drop(['Ende'], axis=1, inplace=True)
    if end==True and new_level==2:
        df.drop(['date_time_end'], axis=1, inplace=True)        
    #cpnvert german float to english 
    if str_convert==True:
        df['residual_power'] =df['residual_power'].astype(str)
        df['residual_power'] = df['residual_power'].str.replace('.','')
        df['residual_power'] = df['residual_power'].str.replace('-','0')
        df['residual_power'] = df['residual_power'].str.replace(',','.').astype(float)/1000.
        df['total_power'] =df['total_power'].astype(str)
        df['total_power'] = df['total_power'].str.replace('.','')
        df['total_power'] = df['total_power'].str.replace('-','0')    
        df['total_power'] = df['total_power'].str.replace(',','.').astype(float)/1000.
        if pump==True:
            df['pump_storage'] =df['pump_storage'].astype(str)
            df['pump_storage'] = df['pump_storage'].str.replace('.','')
            df['pump_storage'] = df['pump_storage'].str.replace(',','.').astype(float)/1000.       
    #old not combined date and time         
    if new_level<2:        
        df['date_time']=pd.to_datetime(df['Date'] + '.' + df['Time'], format='%d.%m.%Y.%H:%M')
    #new is combined     
    elif new_level==2:
        df['date_time']=pd.to_datetime(df['date_time'], format='%d.%m.%Y %H:%M')
    delta=str(df.loc[0,'date_time']-zero)
    #deltam=time.strftime(delta,'%M')
    days=delta.split(' days ')
    hour=days[1].split(':')
    #difference in fraction of days
    diff_frac=float(days[0])+float(hour[0])/24+float(hour[1])/24/60
    time1=np.zeros((df.shape[0],5))
    for i in range(df.shape[0]):
        time1[i,0]=diff_frac+i/4/24
        time1[i,1]=time1[i,0]%1
        time1[i,2]=(time1[i,0]%7)/7
        time1[i,3]=(time1[i,0]%365.25)/365.25
        time1[i,4]=i/4/24/365.25      
    df['frac_day']=time1[:,1]
    df['frac_week']=time1[:,2]
    df['frac_year']=time1[:,3]
    #works at least for full days, later more checks 
    #helper parameter used below
    df['year']=df.date_time.dt.year.astype(int)
    df['month']=df.date_time.dt.month.astype(int)
    #days from easter of this year
    df['delta_easter']=0
    #days from first march of the year, gest holidayts right given leap years
    df['delta_march']=0
    #works at least for full days, later more checks 
    for i in range(df.shape[0]):
        df['delta_easter'].iloc[i]=(df['date_time'].iloc[i]-pd.to_datetime(easter(df.year.iloc[i]))).days
        if df.month.iloc[i]<2.5:
            df['delta_march'].iloc[i]=(df['date_time'].iloc[i]-datetime(df['year'].iloc[i]-1,3,1)).days
        else:
            df['delta_march'].iloc[i]=(df['date_time'].iloc[i]-datetime(df['year'].iloc[i],3,1)).days            
    #exclude what is zero at the end
    c=0
    while df.loc[df.shape[0]+c-1,'total_power']==0:
        c-=1
    #interpolate linear bad value (only exatcly 0 for now) which are not at the beginning or end 
    for i in range(1,df.shape[0]+c-2):
        if df.loc[i,'total_power']==0:
            df.loc[i,'total_power']=(df.loc[i-1,'total_power']+df.loc[i+1,'total_power'])/2    
    #return of the the needed columns in the right order, and dleta when last is likely bad
    delta=0                   
    if df.loc[df.shape[0]+c-1,'total_power']/df.loc[df.shape[0]+c-2,'total_power']<bad_cut:
        delta=-1
    #return needed output, order matters    
    if version=='y_fraction1':    
        return df.loc[:df.shape[0]+c-1+delta,['total_power','frac_day', 'frac_week', 'frac_year','date_time']]
    elif version=='d_easter1':    
        return df.loc[:df.shape[0]+c-1+delta,['frac_day','frac_week','delta_easter','total_power','date_time']] 
    elif version=='d_march1':    
        return df.loc[:df.shape[0]+c-1+delta,['frac_day','frac_week','delta_march','total_power','date_time']]    
