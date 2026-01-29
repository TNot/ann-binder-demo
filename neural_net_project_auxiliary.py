# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 23:12:24 2022

@author: janot
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:45:40 2021

@author: janot
"""

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Dense 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.saved_model import load
try:
    import tensorflow_addons as tfa
except:
    pass
# from keras.models import Sequential
# from keras.models import load_model
# from keras.layers import Activation, Dense 
# from keras.layers import BatchNormalization
# from keras.layers import Dropout


from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import scipy

class MyModel(Sequential):
    
    def __init__(self):
        super().__init__()
    
    def make_me(self, input_size, output_size, size_vector,\
                kernel_initializer_vector, activation_vector, dropout_vector,
                add_BatchNorm_vector, normalize_weights_vector): 
        for ll in range(len(size_vector)):
            if ll == 0:
                if normalize_weights_vector[ll]:
                    self.add(tfa.layers.WeightNormalization(Dense(size_vector[ll], input_shape=(input_size,),
                              activation=activation_vector[ll],
                              kernel_initializer=kernel_initializer_vector[ll])))
                else:
                    self.add(Dense(size_vector[ll], input_shape=(input_size,),
                              activation=activation_vector[ll],
                              kernel_initializer=kernel_initializer_vector[ll]))
            else:
                if normalize_weights_vector[ll]:
                    self.add(tfa.layers.WeightNormalization(Dense(size_vector[ll],
                              activation=activation_vector[ll],
                              kernel_initializer=kernel_initializer_vector[ll])))
                else:
                    self.add(Dense(size_vector[ll],
                              activation=activation_vector[ll],
                              kernel_initializer=kernel_initializer_vector[ll]))
            if add_BatchNorm_vector[ll]:
                self.add(BatchNormalization())
            self.add(Dropout(dropout_vector[ll]))
        
    def compile_me(self, optimizer, loss="mean_absolute_error"):
        self.compile(optimizer=optimizer, loss=loss)
    
    def fit_data(self, input_dat, output_dat, epochs, validation_data=None,\
                 validation_split=0, callbacks=None, batch_size=64):
        history = self.fit(input_dat, output_dat, batch_size=batch_size,
                            epochs=epochs, validation_data=validation_data,
                        validation_split=validation_split, callbacks=callbacks)
        return history
    
    def calculate(self, params):
        prediction = self.predict(np.array([params]))        
        return prediction[0]
    
    
    def evaluate_me(self, input_dat, output_dat, batch_size=64):
        ev = self.evaluate(input_dat, output_dat, batch_size=batch_size)
        return ev
        
    def set_weights_from_loaded_model(self, loadpath=None, loaded_model=None):
        if loadpath is not None:
            loaded = load_model(loadpath)
        else:
            loaded = loaded_model
        weights = loaded.get_weights()
        
        self.set_weights(weights)
        
    def make_model_from_loaded_model(self, optimizer, loaded_model=None,
                                     load_path=None):
        if load_path is not None:
            loaded = load_model(load_path)
        else:
            loaded = loaded_model
            
        config = loaded.get_config()
        
        for ll, layer in enumerate(config['layers']):
            input_layer_made = False
            class_name = layer['class_name']
            layer_config = layer['config']
            if class_name == 'BatchNormalization':
                self.add(BatchNormalization())
            elif class_name == 'Dense':
                if not input_layer_made:
                    input_size = config['layers'][0]['config']\
                                        ['batch_input_shape'][1]
                    self.add(Dense(layer_config['units'],
                                   input_shape=(input_size,) ,
                                   activation=layer_config['activation'],
                        kernel_initializer=layer_config['kernel_initializer']))
                    input_layer_made = True
                elif input_layer_made:
                    self.add(Dense(layer_config['units'], layer_config['activation'],
                        kernel_initializer=layer_config['kernel_initializer']))
                output_size = layer_config['units']
            elif class_name == 'Addons>WeightNormalization':
                inner_layer_config = layer_config['layer']['config']
                if not input_layer_made:
                    input_size = config['layers'][0]['config']\
                                        ['batch_input_shape'][1]
                    self.add(tfa.layers.WeightNormalization(Dense(inner_layer_config['units'],
                                   input_shape=(input_size,) ,
                                   activation=inner_layer_config['activation'],
                        kernel_initializer=inner_layer_config['kernel_initializer'])))
                    input_layer_made = True
                elif input_layer_made:
                    self.add(tfa.layers.WeightNormalization(Dense(inner_layer_config['units'],
                                   activation=inner_layer_config['activation'],
                        kernel_initializer=inner_layer_config['kernel_initializer'])))
                output_size = inner_layer_config['units']
            elif class_name == 'Dropout':
                self.add(Dropout(layer_config['rate']))
        
        self.compile_me(optimizer)


        self.evaluate_me(np.zeros((1,input_size)), np.zeros((1,output_size)),
                         batch_size=64)
        self.set_weights_from_loaded_model(loaded_model=loaded)
        
class Model_averageger:
    
    def __init__(self, models):
        self.models = models
        self.number_of_models = len(models)
    
    def calculate(self, params):
        prediction = 0
        for mm in self.models:
            prediction += mm.calculate(params)
        return prediction/self.number_of_models
    
    def evaluate_me(self, input_dat, output_dat, batch_size=64,
                    loss='mean_absolute_error'):
        loss = 0
        for ii in range(input_dat.shape[0]):
            predicted = self.calculate(input_dat[ii])
            loss += np.average(np.abs(predicted - output_dat[ii]))
        return loss/input_dat.shape[0]
        
            
class Dataloader:
    
    def __init__(self):
        pass
    
    def load_data(self, paths, validation_fname_ending_strings=[],
                  testing_fname_ending_strings=[],\
                  spectra_name_startswith='spectra_batch_',\
                  params_name_startswith='params_batch_', stride=1,
                  maximum_batches=5000,
                  normalise_spectra=False):
        
        params = []
        spectra = []
        val_params = []
        val_spectra = []
        test_params = []
        test_spectra = []
        
        test_spectra_fnames = [] #this is an INELEGANT solution; it'll have to do.
        val_spectra_fnames = []
        test_params_fnames = []
        val_params_fnames = []
        
        for ii, path in enumerate(paths):
            files = os.listdir(path)
            files.sort(key = lambda s: s[s.find('batch_') + 6:].strip('.npy_'))
            
            for ff in files:
                if not ff.endswith('.npy'):
                    continue
                
                if int(ff[ff.find('batch_') + 6:].strip('.npy_')) > maximum_batches:
                    continue
                
                load_path = os.path.join(path, ff)   
                if spectra_name_startswith in ff:
                    for endstring in validation_fname_ending_strings:
                        if ff.endswith(endstring + '.npy'):
                            _dat = np.load(load_path)
                            _dat = _dat[...,::stride]
                            if normalise_spectra:
                                _dat/= np.amax(_dat, axis=-1)[...,None]
                            if np.ndim(_dat) > 1:
                                _dat = np.reshape(_dat, (_dat.shape[0], \
                                          np.prod(_dat.shape[1:])), order='C')  
                            val_spectra.append(_dat)
                            val_spectra_fnames.append(ff)
                    for endstring in testing_fname_ending_strings:
                        if ff.endswith(endstring + '.npy'):
                            _dat = np.load(load_path)
                            _dat = _dat[...,::stride]
                            if normalise_spectra:
                                _dat/= np.amax(_dat, axis=-1)[...,None]
                            if np.ndim(_dat) > 1:
                                _dat = np.reshape(_dat, (_dat.shape[0], \
                                          np.prod(_dat.shape[1:])), order='C')  
                            test_spectra.append(_dat)
                            test_spectra_fnames.append(ff)
                    if ff not in test_spectra_fnames and ff not in val_spectra_fnames:
                        _dat = np.load(load_path)
                        _dat = _dat[...,::stride]
                        if normalise_spectra:
                                _dat/= np.amax(_dat, axis=-1)[...,None]
                        if np.ndim(_dat) > 1:
                            _dat = np.reshape(_dat, (_dat.shape[0], \
                                          np.prod(_dat.shape[1:])), order='C')   
                        spectra.append(_dat)
                elif params_name_startswith in ff:
                    for endstring in validation_fname_ending_strings:
                        if ff.endswith(endstring + '.npy'):
                            val_params.append(np.load(load_path))
                            val_params_fnames.append(ff)
                    for endstring in testing_fname_ending_strings:
                        if ff.endswith(endstring + '.npy'):
                            test_params.append(np.load(load_path))
                            test_params_fnames.append(ff)
                    if ff not in test_params_fnames and ff not in val_params_fnames:
                        params.append(np.load(load_path))
                
        return_ls = [params, spectra, val_params, val_spectra,
                     test_params, test_spectra]
        for ii in range(len(return_ls)):
            if len(return_ls[ii]) > 0:
                return_ls[ii] = np.vstack(return_ls[ii])
            else:
                return_ls[ii] = None
        
        return return_ls
        
            
    def normalise_io(self, params, spectra, val_params=None,
                     val_spectra=None, test_params=None, test_spectra=None,
                     use_params_mean=None, use_params_std=None,
                     use_spectra_mean=None, use_spectra_std=None):
        return_tuple = ()
        
        if use_params_mean is None and use_params_std is None\
            and use_spectra_mean is None and use_spectra_std is None:
            params_mean = np.mean(params, axis=0)
            params_std = np.std(params, axis=0)
            spectra_mean = np.mean(spectra, axis=0)
            spectra_std = np.std(spectra, axis=0)
        else:
            params_mean = use_params_mean
            params_std = use_params_std
            spectra_mean = use_spectra_mean
            spectra_std = use_spectra_std
            
        norm_params = np.where(params_std[None,:] !=0,\
                                      (params - params_mean)/params_std, 0)
        
        
        norm_spec = (spectra - spectra_mean)/spectra_std
        
        return_tuple += (norm_params, norm_spec)
        
        if val_params is not None:
            norm_val_params = np.where(params_std[None,:] !=0,\
                                      (val_params - params_mean)/params_std, 0)
            return_tuple += (norm_val_params,)
            
        if val_spectra is not None:
            norm_val_spectra = (val_spectra \
                                - spectra_mean)/spectra_std  
            return_tuple += (norm_val_spectra,)
        
        if test_params is not None:
            norm_test_params = np.where(params_std[None,:] !=0,\
                                      (test_params - params_mean)/params_std, 0)
            return_tuple += (norm_test_params,)
            
        if test_spectra is not None:
            norm_test_spectra = (test_spectra \
                                - spectra_mean)/spectra_std  
            return_tuple += (norm_test_spectra,)
        
        return return_tuple
    
    def normalise_spec_max(self, params, spectra, val_params=None,
                     val_spectra=None, test_params=None, test_spectra=None):
        
        return_tuple = ()
        
        norm_spec = spectra/(np.amax(spectra, axis=-1))[:,None]
        
        return_tuple += (params, norm_spec)
        
        if val_params is not None:
            return_tuple += (val_params,)
            
        if val_spectra is not None:
            norm_val_spectra = val_spectra/(np.amax(val_spectra, axis=-1))[:,None]
            return_tuple += (norm_val_spectra,)
        
        if test_params is not None:
            return_tuple += (test_params,)
            
        if test_spectra is not None:
            norm_test_spectra = test_spectra/(np.amax(test_spectra, axis=-1))[:,None]
            return_tuple += (norm_test_spectra,)
        
        return return_tuple
    
    def get_mean_and_std(self, data_array):
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)

        return (mean, std)
    

class DataManipulator:
    
    @staticmethod 
    def stat_normalise(data, mean, std):
        return (data - mean)/std
    
    @staticmethod 
    def stat_denormalise(data, mean, std):
        return data*std + mean
    
    @staticmethod 
    def reshape_data(data, new_shape):
        return np.reshape(data, new_shape)
    
    @staticmethod 
    def normalise_traces_individually(data, len_section):
        if len(data.shape) == 2:
            num_sections = data.shape[1]/len_section
            data = np.hstack([dd/((np.amax(dd, axis=1) - np.amin(dd, axis=1))[:,None]) for dd in \
                          np.hsplit(data, num_sections)])
            data = np.hstack([dd - np.amin(dd, axis=1)[:,None] for dd in \
                          np.hsplit(data, num_sections)])
        elif len(data.shape) == 1:
            num_sections = data.size/len_section #in case the data is one-dimensional
            data = np.hstack([dd/(np.amax(dd) - np.amin(dd)) for dd in \
                          np.hsplit(data, num_sections)])
            data = np.hstack([dd - np.amin(dd) for dd in \
                          np.hsplit(data, num_sections)])
        
        return data
    
    @staticmethod 
    def normalise_sections(data, sections_indices):
        if len(data.shape) == 2:
            for section in sections_indices:
                data_section = data[:, section[0]: section[1]]
                data_section /= np.amax(np.abs(data_section), axis=1)[:, None]
                data[:, section[0]: section[1]] = data_section
            
        elif len(data.shape) == 1:
            for section in sections_indices:
                data_section = data[section[0]: section[1]]
                data_section /= np.amax(np.abs(data_section))
                data[section[0]: section[1]] = data_section
        
        return data
    
    @staticmethod 
    def normalise_whole_spectrum(data):
        if len(data.shape) == 2:
            data = data/(np.amax(np.abs(data), axis=-1)[:,None])
        elif len(data.shape) == 1:
            data = data/np.amax(np.abs(data))
        
        return data
    
    @staticmethod
    def select_from_1D_with_2D_indices(data, NDim_shape, indices2D):
        if len(NDim_shape) == 4:
            data = data.reshape(NDim_shape)
            data = data[:, indices2D[0], indices2D[1] ,:]
            data = data.reshape((data.shape[0], np.prod(data.shape[1:])))
        
        return data