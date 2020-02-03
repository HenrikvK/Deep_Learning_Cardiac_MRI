from __future__ import print_function
import csv
import numpy as np
from model import get_model
from utils import real_to_cdf, preprocess
import matplotlib.pyplot as plt

'''
Uses the weights and val loss from the training step to predict the CDF for the separated test set and evaluates its CRPS
score.
It CRPS score is once evaluated for each slice individually and once averaged for all slices across the agent

Input: 
Test data:
data/test/X_test.npy
data/test/scaling_test.npy
data/test/ids_test.npy
data/test/y_test.npy

data/weights_systole_best.hdf5  - best systole weights from training data
data/weights_diastole_best.hdf5  - best diastole weigths from training data
data/val_loss.txt - loss function value of the best weights -> used to calculate cdf

Output:
CRPS score for averaged predictions and for individual Sax slices
'''


def load_validation_data():
    """
    Load validation data from .npy files.
    """
    X = np.load('data/X_validate.npy')
    scaling = np.load('data/scaling_validate.npy')
    
    ids = np.load('data/ids_validate.npy')

    X = X.astype(np.float32)
    X /= 255

    return X, scaling, ids

def load_test_data():
    """
    Load test data from .npy files.
    """
    X = np.load('data/test/X_test.npy')
    scaling = np.load('data/test/scaling_test.npy')
    
    ids = np.load('data/test/ids_test.npy')
    y = np.load('data/test/y_test.npy')             #true test labels!

    #X = X.astype(np.float32)
    #X /= 255

    return X, scaling, ids, y

def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


def submission():
    """
    Generate submission file for the trained models.
    """
    print('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    print('Loading models weights...')
    
    model_systole.load_weights('data/weights_systole_best.hdf5')
    model_diastole.load_weights('data/weights_diastole_best.hdf5')
    
    #model_systole.load_weights('data/weights_systole 2.hdf5')
    #model_diastole.load_weights('data/weights_diastole 2.hdf5')

    # load val losses to use as sigmas for CDF
    with open('data/val_loss.txt', mode='r') as f:
        val_loss_systole = float(f.readline())
        val_loss_diastole = float(f.readline())
    

    print('Loading validation data...')
    X, scaling, ids, y = load_test_data()


    batch_size = 32
    print('Predicting on validation data...')
    pred_systole = model_systole.predict([X, scaling], batch_size=batch_size, verbose=1)
    pred_diastole = model_diastole.predict([X, scaling], batch_size=batch_size, verbose=1)
    


    # real predictions to CDF
    cdf_pred_systole = real_to_cdf(pred_systole, val_loss_systole)
    cdf_pred_diastole = real_to_cdf(pred_diastole, val_loss_diastole)
    
    # CDF for testlabels data (actually a step function)
    cdf_test = real_to_cdf(np.concatenate((y[:, 0], y[:, 1])))
    
    ##############################################
    print("CRPS for not averaged results!")    
    
    # show difference of cdf for prediction and solution:
    fig1 = plt.figure(figsize=(10, 4), dpi=120)
    plt.title("Comparison of cdf for solution and prediction:")
    plt.plot(cdf_pred_systole[1,:], 'r', label = 'cdf systole example prediction')
    plt.plot(cdf_test[1,:], 'b', label = 'cdf systole example solution')
    plt.xlabel('actual volume (mL)')
    plt.ylabel('cumulative density')
    plt.legend()
    plt.show()  
    

    
    # evaluate CRPS on test data
    #CRPS result:
    cprs_notaveraged = 0
    for Sax in range(cdf_pred_systole.shape[0]):
        cprs_notaveraged += np.sum(np.square(cdf_pred_systole[Sax,:] - cdf_test[Sax,:]))
        cprs_notaveraged += np.sum(np.square(cdf_pred_diastole[Sax,:] - cdf_test[int(len(cdf_test)/2) + Sax,:]))
    cprs_notaveraged = cprs_notaveraged / (cdf_pred_systole.shape[0]*600*2)
    print("cprs_notaveraged =" ,cprs_notaveraged)
    
    
    
    
    ##########################################################
    print("CRPS for averaged results!") 
    
    print('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)
    
    print('Accumulating solution...')    
    #print("cdf_test  = ", cdf_test)
    sol_sub_systole = accumulate_study_results(ids, cdf_test[:int(len(cdf_test)/2)])
    sol_sub_diastole = accumulate_study_results(ids, cdf_test[int(len(cdf_test)/2):])
        
    
    print('Evaluating performance...')
    

    
    # show difference of cdf for prediction and solution:
    fig2 = plt.figure(figsize=(10, 4), dpi=120)
    plt.title("Comparison of cdf for solution and averaged prediction:")
    plt.plot(np.array(sub_systole[1]).flatten(), 'r', label = 'cdf systole patient 1 prediction')
    plt.plot(np.array(sol_sub_systole[1]).flatten(), 'b', label = 'cdf systole patient 1 solution')
    plt.xlabel('actual volume (mL)')
    plt.ylabel('cumulative density')
    plt.legend()
    plt.show()  
    
    
    # evaluate CRPS on test data
    #CRPS result:
    cprs = 0
    for patient in sol_sub_diastole:
        cprs += np.sum(np.square(sol_sub_systole[patient] - sub_systole[patient]))
        cprs += np.sum(np.square(sol_sub_diastole[patient] - sub_diastole[patient]))
    cprs = cprs / (len(sol_sub_diastole)*2*600)
    print("cprs =" ,cprs)
    #####################################################################
    
    # show difference of cdf for prediction and solution:
    fig3 = plt.figure(figsize=(10, 4), dpi=120)
    plt.title("Comparison of cdf for solution and 1 sax slice prediction and the averaged prediction:")
    plt.plot(cdf_pred_systole[1,:], 'g', label = 'cdf systole patient 1 Sax 1 prediction')
    plt.plot(np.array(sub_systole[1]).flatten(), 'r', label = 'cdf systole patient 1 averaged prediction')
    plt.plot(np.array(sol_sub_systole[1]).flatten(), 'b', label = 'cdf systole patient 1 solution')
    plt.xlabel('actual volume (mL)')
    plt.ylabel('cumulative density')
    plt.legend()
    plt.show() 

    print('Done.')

submission()
