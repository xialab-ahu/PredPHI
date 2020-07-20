# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model

def scores(y_test, y_pred, th=0.5):           
    y_predlabel = [(0. if item < th else 1.) for item in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    fpr,tpr,threshold = roc_curve(y_test, y_predlabel)
    sen, spe, pre, f1, mcc, acc, auc, tn, fp, fn, tp = np.array([recall_score(y_test, y_predlabel), SPE, precision_score(y_test, y_predlabel,average='macro'), 
                                                                 f1_score(y_test, y_predlabel), MCC, accuracy_score(y_test, y_predlabel), 
                                                                 roc_auc_score(y_test, y_pred), tn, fp, fn, tp])
    return sen, spe, pre, f1, mcc, acc,auc,tn,fp,fn,tp  

def list_of_groups(init_list, childern_list_len):
    list_of_group = zip(*(iter(init_list),) *childern_list_len)
    end_list = [list(i) for i in list_of_group]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list

def obtainfeature(feature_new_kmeans):
    feature_phage_kmeans=feature_new_kmeans[:,:int(feature_new_kmeans.shape[1]/2)]
    feature_host_kmeans=feature_new_kmeans[:,int(feature_new_kmeans.shape[1]/2):]
    ##obtain phage features
    feature_phage_kmeans_CHONS=[]
    feature_phage_kmeans_weight=[]
    feature_phage_kmeans_AAC=[]
    for i in feature_phage_kmeans:
        cc=i[:5]  #obtain CHONS
        ww=[]    #obtain weight
        ww.append(i[5])  
        aa=i[6:27]  #obtain AAC
        for ii in range(27,len(i),27):   ##obtain mean,max,min...
            cc=np.concatenate((cc,i[ii:ii+5]))
            ww.append(i[ii+5])
            aa=np.concatenate((aa,i[ii+6:ii+27]))
        feature_phage_kmeans_CHONS.append(cc.tolist())
        feature_phage_kmeans_weight.append(ww)
        feature_phage_kmeans_AAC.append(aa.tolist())
    ##obtain host features
    feature_host_kmeans_CHONS=[]
    feature_host_kmeans_weight=[]
    feature_host_kmeans_AAC=[]
    for i in feature_host_kmeans:
        cc=i[:5]
        ww=[]
        ww.append(i[5])
        aa=i[6:27]
        for ii in range(27,len(i),27):
            cc=np.concatenate((cc,i[ii:ii+5]))
            ww.append(i[ii+5])
            aa=np.concatenate((aa,i[ii+6:ii+27]))
        feature_host_kmeans_CHONS.append(cc.tolist())
        feature_host_kmeans_weight.append(ww)
        feature_host_kmeans_AAC.append(aa.tolist())
    ##combine phage and host features
    feature_kmeans_CHONS=np.concatenate((feature_phage_kmeans_CHONS,feature_host_kmeans_CHONS),axis=1)
    feature_kmeans_AAC=np.concatenate((feature_phage_kmeans_AAC,feature_host_kmeans_AAC),axis=1)  
    feature_kmeans_CHONS_AAC=np.concatenate((feature_kmeans_CHONS,feature_kmeans_AAC),axis=1)
    feature_kmeans_CHONS_AAC_new=[]
    for i in range(len(feature_kmeans_CHONS_AAC)):
        feature_kmeans_CHONS_AAC_new.append([feature_kmeans_CHONS_AAC[i,:int(feature_kmeans_CHONS_AAC.shape[1]/2)].reshape((6,-1)),
                                        feature_kmeans_CHONS_AAC[i,int(feature_kmeans_CHONS_AAC.shape[1]/2):].reshape((6,-1))])
    return feature_kmeans_CHONS_AAC_new
def main(feature_test,label_test): 
    model = load_model('../result/model.h5') 
    test_X=np.array(feature_test).transpose(0,2,3,1)
    print(scores(label_test,model.predict(test_X)[:,1]))
    
test_kmeans=pd.read_csv('../data/test_kmeans.csv',header=None,sep=' ').get_values()
feature_new_kmeans=test_kmeans[:,:-1]
label_new_kmeans=test_kmeans[:,-1]
feature_kmeans_CHONS_AAC_new=obtainfeature(feature_new_kmeans)
main(feature_kmeans_CHONS_AAC_new,label_new_kmeans)

test_random=pd.read_csv('../data/test_random.csv',sep=',')
feature_test=[]
label_test=[]
for i in test_random.index:
    dd_test=pd.read_csv('../data/testfeatures/'+str(test_random.ix[i,'phage'])+','+str(test_random.ix[i,'host'])+
                        '.csv',sep='\t',header=None).get_values()[:,0]
    feature_test.append(dd_test.T.tolist())
    label_test.append(test_random.ix[i,'class']) 
label_new_test=np.array(label_test)
feature_random_CHONS_AAC=obtainfeature(np.array(feature_test))
main(feature_random_CHONS_AAC,label_test)

