import numpy as np
import random

from collections import namedtuple
import numpy.random as rd
import tensorflow.compat.v1 as tf
from tensorflow.python.ops.variables import Variable

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore",category=FutureWarning)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.disable_v2_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #use CPU for this experiment

#from neuron_model.ALIF import ALIF
#from neuron_model.LIF import ALIF
from neuron_model.AdEx import ALIF 

#######################################################################################
# model fitting experiment
#######################################################################################

def experiment(cell, input_file, peak_high, nonspike_fig):
    
    mse_error = 0
    peak_error = 0
    voltageList, currentList = [], []
                                                                                        
    time_step = 200   
    n_batch = 1       
    n_in = 1
    noSpike_error = 100
    mse_scale = 10
    
    # run simulation
    inputs = tf.placeholder(tf.float32, [n_batch, time_step, n_in], name='inputs')
    init_state = cell.zero_state(n_batch, tf.float32)
    rnn_out, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
    session = tf.Session()
    session.run(tf.global_variables_initializer())  
    for current in range(25, 300, 25): #current range, from 25pA to 300pA  
        
        sample = np.zeros(time_step)
        sample[50:150] = [current]*100
        in_current = np.reshape(sample, (n_batch, time_step, n_in) )
        out = session.run([rnn_out], feed_dict={inputs: in_current})
        out = np.array(out[0])   # [u, s, w, I1, I2, out], n_in, timestep, neuron

        voltageList.append(out[0][0].T[0])
        currentList.append(current)
    
    # readin experimental data
    targetList = np.loadtxt(input_file)
    targetList = targetList.reshape(11, -1)
    targetList = targetList[:,::10]  #match sampling rate                                                                                        
    voltageList = np.array(voltageList).reshape(11,-1)
    currentList = np.array(currentList).reshape(11,-1)

    # calculate error
    for i in range(11,12): #1, 12
        peak_p = find_peaks(voltageList[i-1], height=peak_high)[0]
        peak_t = find_peaks(targetList[i-1], height=peak_high)[0]
        length = min(len(peak_p), len(peak_t)) #number of spikes
        diff = peak_p[:length]-peak_t[:length]
        diff = np.append(diff, [noSpike_error]*(max(len(peak_p), len(peak_t))-length)) #error
        weight = np.arange(1, 10)[::-1]
        weight[2:]=1
        if len(weight)>len(diff):
                weight = weight[:len(diff)] 
        else:
                weight = np.append(weight, [1]*(len(diff)-len(weight)))

        #print(peak_p, peak_t, diff, weight)
                                                                                        
        if len(diff):
            peak_error += sum(map(abs,diff*weight))

        if i<nonspike_fig:
            mse_error+=((voltageList[i-1] - targetList[i-1])**2).mean(axis=0)*mse_scale

    print("mse_error/peak_error", mse_error, peak_error)
    return peak_error+mse_error
    
#######################################################################################
# optimization with OPTUNA
#######################################################################################
import optuna
from scipy.signal import find_peaks

def optimization(trial):
	dt=1 #1 ms
	n_in, n_rec, n_out = 1, 1, 1 
	inW=np.ones([n_in, n_rec]) 
	inBias=np.zeros([n_in, n_rec])
	recW=np.zeros([n_rec, n_rec])
	outW = np.zeros([n_rec, n_out])	
	
	param={'Vahp':0,'Vth':0,'Vmax':0,'t_ref':0,'R':0,'tau_m':0,'R_adp':0, \
				 'tau_w':0,'a':0,'delT':0,'p':0,'q':0,'r':0,'s':0}
	param['Vahp'] =  trial.suggest_float('Vahp', -60, -15, step=0.5)
	param['Vth'] = trial.suggest_float('Vth', -60, -15, step=0.5)
	param['Vmax'] =  trial.suggest_float('Vmax', 10, 40, step=0.5)
	param['t_ref']  =  trial.suggest_float('t_ref', 0.5, 20, step=0.5)
	param['R'] = trial.suggest_float('R', 0.1, 1, step=0.1)
	param['tau_m'] = trial.suggest_float('tau_m', 0.5, 100, step=0.5)
	
	param['R_adp']  =  trial.suggest_float('R_adp', 0.1, 1, step=0.1)
	param['tau_w']  =  trial.suggest_float('tau_w', 0.5, 100, step=0.5)
	param['a'] =  trial.suggest_float('a', 0.5, 100, step=0.5)
	param['delT'] = trial.suggest_float('delT', 0.5, 10, step=0.5)
	param['p'] = trial.suggest_float('p', 0.1, 10, step=0.1)
	param['q'] = trial.suggest_float('q', 0.1, 10, step=0.1)
	param['r'] = trial.suggest_float('r', 0.1, 10, step=0.1)
	param['s'] = trial.suggest_float('s', 0.1, 10, step=0.1)
	
	neuron_param1 = [-58.5, param['Vth'], param['Vahp'], param['Vmax']]  #Vrest, Vth, Vaph, Vmax 
	neuron_param2 = [param['t_ref']*dt, param['R'], param['tau_m']*dt] #t_ref, R, tua_m
	neuron_param3 = [param['R_adp'], param['tau_w']*dt, param['a']]   #R_adp, tau_w, a 
	
	network_param = [n_in, n_rec, n_out]  #n_in, n_rec, n_out
	synapse_param = [100*dt, 100*dt, inW, inBias, recW, outW]  #tau_recsyn, tau_postsyn, inW, inBias, recW, outW
	cell = ALIF(network_param, neuron_param1+neuron_param2+neuron_param3, synapse_param)

	input_file = 'data_ex.txt' #experiment data 
	peak_high = 10  #used to recognize spike for experiment curves 
	nonspike_fig = 3 #number of curves that do not have spikes
	total_error = experiment(cell, input_file, peak_high, nonspike_fig)
	
	return total_error

#######################################################################################
study = optuna.create_study(direction='minimize')
study.optimize(optimization, n_trials=1000)

