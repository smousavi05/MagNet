#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:47:28 2019

@author: mostafamousavi
"""
from keras.layers import add, ConvLSTM2D, Reshape, Dense, AveragePooling2D, Input, Conv2DTranspose, TimeDistributed, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import add, Reshape, Dense, Input, TimeDistributed, Dropout, Activation, LSTM, Conv1D, Cropping1D
from keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from keras.models import Model, Sequential
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras import metrics
from keras.optimizers import Adam
import locale
import os
import matplotlib

#matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from os import listdir, walk
from os.path import isfile, join, isdir
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
from datetime import datetime
from datetime import timedelta
import os.path
from keras import Sequential
from keras.layers import Dense
import h5py
import tensorflow as tf
from keras import backend as K
import pandas as pd
import csv
    

def datat_reader(file_name, file_list):
    net_code = []
    rec_type = []
    eq_id = []
    eq_depth = []
    eq_mag = []
    mag_type = []
    mag_auth = []
    eq_dist = []
    snr = []
    trace_name = []
    S_P = []
    baz = []
            
    dtfl = h5py.File(file_name, 'r')
    X = np.zeros([len(file_list), 3000, 3]) 
    X2 = np.zeros([len(file_list), 1])  
    Y = np.zeros([len(file_list), 1])
   
    pbar = tqdm(total=len(file_list)) 
    for c, evi in enumerate(file_list):
        pbar.update()
        dataset = dtfl.get('earthquake/local/'+str(evi))   
        data = np.array(dataset)
        BAZ = round(float(dataset.attrs['back_azimuth_deg']), 2)
        spt = int(dataset.attrs['p_arrival_sample'])
        sst = int(dataset.attrs['s_arrival_sample'])
        dpt = dataset.attrs['source_depth_km']
        mag = round(float(dataset.attrs['source_magnitude']), 2)
        dis = round(float(dataset.attrs['source_distance_deg']), 2)
        SNR = dataset.attrs['snr_db']
        sp = int(sst - spt)
       
        dshort = data[spt-100:spt+2900, :] 
        X[c, :, :] = dshort 
        Y[c, 0] = mag 
           
        net_code.append(dataset.attrs['network_code'])
        rec_type.append(dataset.attrs['receiver_type'])
        eq_id.append(dataset.attrs['source_id'])
        eq_depth.append(dpt)  
        eq_mag.append(mag)
        mag_type.append(dataset.attrs['source_magnitude_type'])
        mag_auth.append(dataset.attrs['source_magnitude_author'])
        eq_dist.append(dis) 
        snr.append(round(np.mean(SNR), 2))
        trace_name.append(dataset.attrs['trace_name'])
        S_P.append(sp)
        baz.append(BAZ) 
        
    dtfl.close()
                     
    return X, X2, Y, net_code, rec_type, eq_id, eq_depth, eq_mag, mag_type, mag_auth, eq_dist, snr, trace_name, S_P, baz




def string_convertor(dd):
    
    dd2 = dd.split()
    SNR = []
    for i, d in enumerate(dd2):
        if d != '[' and d != ']':
            
            dL = d.split('[')
            dR = d.split(']')
            
            if len(dL) == 2:
                dig = dL[1]
            elif len(dR) == 2:
                dig = dR[0]
            elif len(dR) == 1 and len(dR) == 1:
                dig = d
            try:
                dig = float(dig)
            except Exception:
                dig = None
                
            SNR.append(dig)
    return(SNR)
      


    
###############################################################################
file_name = "./dataset/waveforms.hdf5"
csv_file = "./dataset/metadata.csv" 


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
K.tensorflow_backend.set_session(tf.Session(config=config))

epochs_number = 200
bach_size = 256
monte_carlo_sampling = 50
drop_rate = 0.2

df = pd.read_csv(csv_file) 
df = df[df.trace_category == 'earthquake_local']
df = df[df.source_distance_km <= 110]
df = df[df.source_magnitude_type == 'ml']
df = df[df.p_arrival_sample >= 200]
df = df[df.p_arrival_sample+2900 <= 6000]
df = df[df.p_arrival_sample <= 1500]
df = df[df.s_arrival_sample >= 200]
df = df[df.s_arrival_sample <= 2500]
df = df[df.coda_end_sample <= 3000]
df = df[df.p_travel_sec.notnull()]
df = df[df.p_travel_sec > 0]
df = df[df.source_distance_km.notnull()]
df = df[df.source_distance_km > 0]
df = df[df.source_depth_km.notnull()]
df = df[df.source_magnitude.notnull()]
df = df[df.back_azimuth_deg.notnull()]
df = df[df.back_azimuth_deg > 0]
df.snr_db = df.snr_db.apply(lambda x: np.mean(string_convertor(x)))
df = df[df.snr_db >= 20]

uniq_ins = df.receiver_code.unique()

labM = []
for ii in range(0, len(uniq_ins)):
    print(str(uniq_ins[ii]), sum(n == str(uniq_ins[ii]) for n in df.receiver_code))
    stn = sum(n == str(uniq_ins[ii]) for n in df.receiver_code)
    if stn >= 400:
        labM.append(str(uniq_ins[ii]))
    
    
np.save('multi_observations', labM)

multi_observations = np.load('multi_observations.npy')
ev_list = []
for index, row in df.iterrows():
    st = row['receiver_code']
    if st in multi_observations:
        ev_list.append(row['trace_name'])

ev_list = df.trace_name.tolist()
np.random.shuffle(ev_list)  

training = ev_list[:int(0.7*len(ev_list))]
validation =  ev_list[int(0.7*len(ev_list)): int(0.8*len(ev_list))]  
test =  ev_list[ int(0.8*len(ev_list)):]

model_name = 'mag_regressionLSTM_fullML_multiobservations_nogenerator_longer'
save_dir = os.path.join(os.getcwd(), str(model_name)+'_outputs')
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)  
os.makedirs(save_dir)

x_train, x2_train, y_train, _, _, _, _, _, _, _, _, _, trace_name_train, _, baz_train = datat_reader(file_name, training) 
x_test, x2_test, y_test, net_code, rec_type, eq_id, eq_depth, eq_mag, mag_type, mag_auth, eq_dist, snr, trace_name, S_P, baz_test = datat_reader(file_name, test)  
 

assert not np.any(np.isnan(x_train).any())
assert not np.any(np.isnan(x_test).any())
assert not np.any(np.isnan(x2_train).any())
assert not np.any(np.isnan(x2_test).any())
assert not np.any(np.isnan(y_train).any())
assert not np.any(np.isnan(y_test).any())


filters = [32, 64, 96, 128, 256] 

inp1 = Input(shape=(3000, 3), name='input_layer') 

e = Conv1D(filters[1], 3, padding = 'same')(inp1) 
e = Dropout(drop_rate)(e, training=True)
e = MaxPooling1D(4, padding='same')(e)
 
e = Conv1D(filters[0], 3, padding = 'same')(e) 
e = Dropout(drop_rate)(e, training=True)
e = MaxPooling1D(4, padding='same')(e)

e = Bidirectional(LSTM(100, return_sequences=False, dropout=0.0, recurrent_dropout=0.0))(e)

e = Dense(2)(e)
o = Activation('linear', name='output_layer')(e)

model = Model(inputs=[inp1], outputs=o)
    

#costom loss for calculating aleatoric uncertainty
def customLoss(yTrue, yPred):
    y_hat = K.reshape(yPred[:, 0], [-1, 1]) 
    s = K.reshape(yPred[:, 1], [-1, 1])
    return tf.reduce_sum(0.5 * K.exp(-1 * s) * K.square(K.abs(yTrue - y_hat)) + 0.5 * s, axis=1)
         
model.compile(optimizer='Adam', loss=customLoss)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown= 0,
                                patience= 4,
                                min_lr=0.5e-6)

m_name = str(model_name)+'_{epoch:03d}.h5' 
filepath = os.path.join(save_dir, m_name)

early_stopping_monitor = EarlyStopping(monitor= 'val_loss', patience = 5)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             mode = 'auto',
                             verbose=1,
                             save_best_only=True)

class PrintSomeValues(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print(f'y_test[0:1] = {y_test[0:1]}.')
        print(f'pred = {self.model.predict(x_test[0:1])}.')
        
psv = PrintSomeValues()

callbacks = [lr_reducer, early_stopping_monitor, checkpoint, psv]

history = model.fit(x_train, y_train, epochs=epochs_number, validation_split=0.1, batch_size=bach_size, callbacks = callbacks)


#np.save(save_dir+'/history',history)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'], '--')
ax.legend(['loss', 'val_loss'], loc='upper right') 
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 



######################## TEST

# for calculating the epistemic uncertainty 
class KerasDropoutPrediction(object):
    def __init__(self, model):
        self.model = model
        
    def predict(self, x, n_iter=10):
        predM = []
        auM = []
        
        for itr in range(n_iter):
            print('Prediction: #'+ str(itr+1))
            r = model.predict(x, batch_size=bach_size, verbose=0)
            
            pred = r[:, 0] 
            au = r[:, 1] 
            predM.append(pred.T)
            auM.append(au.T)
        
        predM = np.array(predM).reshape(n_iter,len(predM[0]))
        auM = np.array(auM).reshape(n_iter, len(auM[0])) 
        
        yhat_mean = predM.mean(axis=0)
        yhat_squared_mean = np.square(predM).mean(axis=0)
        
        sigma_squared = 10**(auM) 
        sigma_squared_mean = sigma_squared.mean(axis=0)
        
        ep_unc = predM.std(axis=0)  
        
        combibed = yhat_squared_mean - np.square(yhat_mean)+ sigma_squared_mean
        
        return yhat_mean, sigma_squared_mean, ep_unc, combibed
    
kdp = KerasDropoutPrediction(model)
predic, al_unc, ep_unc, comb = kdp.predict(x_test, monte_carlo_sampling)


fig1 = plt.figure()
plt.errorbar(predic, al_unc, xerr= al_unc, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test, al_unc, 'ro', alpha=0.4)
plt.xlabel('Magnitude')
plt.ylabel('Aleatoric Uncertainty')
plt.title('Aleatoric Uncertainty')
fig1.savefig(os.path.join(save_dir,'plots1.png')) 

fig2 = plt.figure()
plt.errorbar(predic, ep_unc, xerr= ep_unc, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test, ep_unc, 'ro', alpha=0.4)
plt.xlabel('Magnitude')
plt.ylabel('Epistemic Uncertainty')
plt.title('Epistemic Uncertainty')
fig2.savefig(os.path.join(save_dir,'plots2.png')) 

fig3 = plt.figure()
plt.errorbar(predic, al_unc, xerr= comb, fmt='o', alpha=0.4, ecolor='g', capthick=2)
plt.plot(y_test, al_unc, 'ro', alpha=0.4)
plt.xlabel('Magnitude')
plt.ylabel('Combined Uncertainty')
plt.title('Combined Uncertainty')
fig3.savefig(os.path.join(save_dir,'plots3.png')) 

fig4, ax = plt.subplots()
ax.scatter(y_test, predic, alpha = 0.4, facecolors='none', edgecolors='r')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.4, lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig4.savefig(os.path.join(save_dir,'plots4.png')) 



csvfile = open(os.path.join(save_dir,'results.csv'), 'w')          
output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
output_writer.writerow(['network_code', 'reciever_type', 'source_id', 'source_depth_km', 'source_magnitude',
                        'mag_type', 'mag_auth', 'eq_dist', 'snr', 'trace_name', 'S-P',  
                        'true_magnitude', 'predicted_magnitude', 'diff', 'al_uncertainty', 'ep_uncertainty', 'comb_uncertainty'])
csvfile.flush()

#predic = model.predict(x_test)


diff = []
for i, v in enumerate(y_test):
    pre = round(predic[i], 2)
    d = round((v[0] - pre), 3)
    diff.append(d)

    output_writer.writerow([net_code[i], 
                            rec_type[i], 
                            eq_id[i], 
                            eq_depth[i], 
                            eq_mag[i], 
                            mag_type[i], 
                            mag_auth[i], 
                            eq_dist[i], 
                            snr[i],
                            trace_name[i],
                            S_P[i], 
                            eq_mag[i], 
                            pre, 
                            d,
                            round(al_unc[i], 3),
                            round(ep_unc[i], 3),
                            round(comb[i], 3),                           
                            ])           
    csvfile.flush() 



print('Writting results into: " ' + str(model_name)+' "')

with open(os.path.join(save_dir,'report.txt'), 'a') as the_file:
    the_file.write('file_name: '+str(file_name)+'\n')
    the_file.write('model_name: '+str(model_name)+'\n')    
    the_file.write('epoch_number: '+str(epochs_number,)+'\n')
    the_file.write('total number of training: '+str(len(x_test))+'\n')
    the_file.write('total number of test: '+str(len(x_test))+'\n')
    the_file.write('average error: '+str(np.round(np.mean(diff), 2))+'\n')
    the_file.write('average error_std: '+str(np.round(np.std(diff), 2))+'\n')  
    the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
    the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
    the_file.write('monte_carlo_sampling: '+str(monte_carlo_sampling)+'\n')
    the_file.write('dropout_rate: '+str(drop_rate)+'\n')
    the_file.write('mean combination error: '+str(np.mean(comb))+'\n')
    the_file.write('mean Aleatoric Uncertainty: '+str(np.mean(al_unc))+'\n')
    the_file.write('mean Epistemic Uncertainty: '+str(np.mean(ep_unc))+'\n')





class KerasDropoutPrediction(object):
    def __init__(self, model):
        self.model = model
        
    def predict(self, x, n_iter=10):
        predM = []
        auM = []
        
        for itr in range(n_iter):
            print('Prediction: #'+ str(itr+1))
            r = model.predict(x, batch_size=bach_size, verbose=0)
            
            pred = r[:, 0] 
            au = r[:, 1] 
            predM.append(pred.T)
            auM.append(au.T)
        
        predM = np.array(predM).reshape(n_iter,len(predM[0]))
        auM = np.array(auM).reshape(n_iter, len(auM[0])) 
        
        yhat_mean = predM.mean(axis=0)
        yhat_squared_mean = np.square(predM).mean(axis=0)
        
        sigma_squared = 10**(auM) 
        sigma_squared_mean = sigma_squared.mean(axis=0)
        
        ep_unc = predM.std(axis=0)  
        
        combibed = yhat_squared_mean - np.square(yhat_mean)+ sigma_squared_mean
        
        return yhat_mean, sigma_squared_mean, ep_unc, combibed
    
kdp = KerasDropoutPrediction(model)
predic, al_unc, ep_unc, comb = kdp.predict(x_train, monte_carlo_sampling)

fig4, ax = plt.subplots()
ax.scatter(y_train, predic, alpha = 0.4, facecolors='none', edgecolors='r')
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', alpha=0.4, lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig4.savefig(os.path.join(save_dir,'plots5.png')) 