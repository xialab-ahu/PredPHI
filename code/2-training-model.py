# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score, matthews_corrcoef,roc_curve
import plaidml.keras
import keras.applications
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import matplotlib
matplotlib.use("Agg")
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import Callback
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.callbacks import EarlyStopping

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_recall = recall_score(val_targ, val_predict,average=None)
        _val_precision = precision_score(val_targ, val_predict,average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return _val_f1


class PredPHI:
    def build(self,width, height, depth, classes, finalAct="softmax"):
        model = Sequential()
        inputShape = (width, height, depth)
        chanDim = -1
        
        if(K.image_data_format == "channels_first"):
            inputShape = (depth, height, width)  
            chanDim = 1

        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.5))
              
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation(finalAct))
        
        return model

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

def obtainfeature(data):
    feature=data[:,:-1]
    label=data[:,-1]
    feature_phage=feature[:,:int(feature.shape[1]/2)]
    feature_host=feature[:,int(feature.shape[1]/2):]
    
    ##obtain phage features
    feature_phage_CHONS=[]
    feature_phage_weight=[]
    feature_phage_AAC=[]
    for i in feature_phage:
        cc=i[:5]  #obtain CHONS
        ww=[]    #obtain weight
        ww.append(i[5])  
        aa=i[6:27]  #obtain AAC
        for ii in range(27,len(i),27):   ##obtain mean,max,min...
            cc=np.concatenate((cc,i[ii:ii+5]))
            ww.append(i[ii+5])
            aa=np.concatenate((aa,i[ii+6:ii+27]))
        feature_phage_CHONS.append(cc.tolist())
        feature_phage_weight.append(ww)
        feature_phage_AAC.append(aa.tolist())
    ##obtain host features
    feature_host_CHONS=[]
    feature_host_weight=[]
    feature_host_AAC=[]
    for i in feature_host:
        cc=i[:5]
        ww=[]
        ww.append(i[5])
        aa=i[6:27]
        for ii in range(27,len(i),27):
            cc=np.concatenate((cc,i[ii:ii+5]))
            ww.append(i[ii+5])
            aa=np.concatenate((aa,i[ii+6:ii+27]))
        feature_host_CHONS.append(cc.tolist())
        feature_host_weight.append(ww)
        feature_host_AAC.append(aa.tolist())
    ##combine phage and host features
    feature_CHONS=np.concatenate((feature_phage_CHONS,feature_host_CHONS),axis=1)
    feature_AAC=np.concatenate((feature_phage_AAC,feature_host_AAC),axis=1)  
    feature_CHONS_AAC=np.concatenate((feature_CHONS,feature_AAC),axis=1)
    feature_CHONS_AAC_new=[]
    for i in range(len(feature_CHONS_AAC)):
        feature_CHONS_AAC_new.append([feature_CHONS_AAC[i,:int(feature_CHONS_AAC.shape[1]/2)].reshape((6,-1)),
                                        feature_CHONS_AAC[i,int(feature_CHONS_AAC.shape[1]/2):].reshape((6,-1))])
    return feature_CHONS_AAC_new,label
def main(data): 
    INIT_LR=1e-4
    BS=32
    EPOCHS = 300
    FEATURE_DIMS = (6,26,2) 
    ###kfold result
    kf = KFold(n_splits=10,random_state=1)
    result_predphi=[]
    predphi_pred=[]
    test_y_all=[]
    for train_index, test_index in kf.split(data): 
        training=data[train_index,:]
        test=data[test_index,:]   
        feature_training,label_training=obtainfeature(training)
        feature_test,label_test=obtainfeature(test)
        feature_training2=np.array(feature_training).transpose(0,2,3,1)
        label_training2 = keras.utils.to_categorical(label_training)
        feature_test2=np.array(feature_test).transpose(0,2,3,1)
        test_y_all=test_y_all+label_test 
        model = PredPHI().build(width=FEATURE_DIMS[0], height=FEATURE_DIMS[1], depth=FEATURE_DIMS[2], classes=2)
        opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['acc'])
        f1=Metrics()
        model.fit(feature_training2, label_training2, batch_size=BS,epochs=EPOCHS,validation_data=(feature_training2, label_training2), verbose=1,
                      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto'),f1])
        result_predphi.append(scores(label_test,model.predict(feature_test2)[:,1]))
        predphi_pred=predphi_pred+model.predict(feature_test2)[:,1].tolist()  
    np.savetxt('../result/pred-10fold.csv',np.array([test_y_all,predphi_pred]).T)
    result=np.mean(result_predphi,axis=0)
    with open('../result/result_score-10fold.csv','w') as fout:
        fout.write('sen,spe,pre,f1,mcc,acc,auc,tn,fp,fn,tp\n')
        for jj in result:
            fout.write(str(jj)+',')
        fout.write('\n')
    
        
    ###construct model  
    data=pd.read_csv('../data/training_kmeans.csv',sep='\t')
    feature,label=obtainfeature(data)
    feature2=np.array(feature).transpose(0,2,3,1)
    label2 = keras.utils.to_categorical(label)
    model = PredPHI().build(width=FEATURE_DIMS[0], height=FEATURE_DIMS[1], depth=FEATURE_DIMS[2], classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['acc'])
    f1=Metrics()
    model.fit(feature2, label2, batch_size=BS,epochs=EPOCHS,validation_data=(feature2, label2), verbose=1,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto'),f1])
    mp = '../result/model.h5'
    model.save(mp)

data=pd.read_csv('../data/training_kmeans.csv',header=None,sep=' ').get_values()
#main(data)
INIT_LR=1e-4
BS=32
EPOCHS = 300
FEATURE_DIMS = (6,26,2) 
###kfold result
kf = KFold(n_splits=10,random_state=1)
result_predphi=[]
predphi_pred=[]
test_y_all=[]
for train_index, test_index in kf.split(data): 
    training=data[train_index,:]
    test=data[test_index,:]   
    feature_training,label_training=obtainfeature(training)
    feature_test,label_test=obtainfeature(test)
    feature_training2=np.array(feature_training).transpose(0,2,3,1)
    label_training2 = keras.utils.to_categorical(label_training)
    feature_test2=np.array(feature_test).transpose(0,2,3,1)
    test_y_all=test_y_all+label_test.tolist()
    model = PredPHI().build(width=FEATURE_DIMS[0], height=FEATURE_DIMS[1], depth=FEATURE_DIMS[2], classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['acc'])
    f1=Metrics()
    model.fit(feature_training2, label_training2, batch_size=BS,epochs=EPOCHS,validation_data=(feature_training2, label_training2), verbose=1,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto'),f1])
    result_predphi.append(scores(label_test,model.predict(feature_test2)[:,1]))
    predphi_pred=predphi_pred+model.predict(feature_test2)[:,1].tolist()  
np.savetxt('../result/pred-10fold.csv',np.array([test_y_all,predphi_pred]).T)
result=np.mean(result_predphi,axis=0)
with open('../result/result_score-10fold.csv','w') as fout:
    fout.write('sen,spe,pre,f1,mcc,acc,auc,tn,fp,fn,tp\n')
    for jj in result:
        fout.write(str(jj)+',')
    fout.write('\n')

    
###construct model  
data=pd.read_csv('../data/training_kmeans.csv',sep=' ',header=None).get_values()
feature,label=obtainfeature(data)
feature2=np.array(feature).transpose(0,2,3,1)
label2 = keras.utils.to_categorical(label)
model = PredPHI().build(width=FEATURE_DIMS[0], height=FEATURE_DIMS[1], depth=FEATURE_DIMS[2], classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['acc'])
f1=Metrics()
model.fit(feature2, label2, batch_size=BS,epochs=EPOCHS,validation_data=(feature2, label2), verbose=1,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2, mode='auto'),f1])
mp = '../result/model.h5'
model.save(mp)


























