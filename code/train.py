from __future__ import print_function
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation

def create_train_test_sets():
    # split available data into train and test, so that models are trained and tested on the same set
    # only run once, manually before training

    X = np.load('data/full/X_train_cropped.npy')
    X = X.astype(np.float32)
    X /= 255
    print('Pre-processing images...')
    X = preprocess(X)

    scaling = np.load('data/full/pixel_spacing_train.npy')
    ids = np.load('data/full/ids_train.npy')
    y = np.load('data/full/y_train_cropped.npy')

    X_train, scaling_train, ids_train, y_train, X_test, scaling_test, ids_test, y_test = split_data(X, scaling, ids, y, split_ratio=0.15) # create test set for final evaluation

    # save data to files
    np.save('data/train/X_train', X_train)
    np.save('data/train/scaling_train', scaling_train)
    np.save('data/train/ids_train', ids_train)
    np.save('data/train/y_train', y_train)

    np.save('data/test/X_test', X_test)
    np.save('data/test/scaling_test', scaling_test)
    np.save('data/test/ids_test', ids_test)    
    np.save('data/test/y_test', y_test)    

def load_train_data():
    """
    Load training data from .npy files.
    """
    X_train = np.load('data/train/X_train.npy')
    scaling_train = np.load('data/train/scaling_train.npy')
    ids_train = np.load('data/train/ids_train.npy')
    y_train = np.load('data/train/y_train.npy')

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(scaling_train)
    np.random.seed(seed)
    np.random.shuffle(ids_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    return X_train, scaling_train, ids_train, y_train

def load_test_data():
    """
    Load test data from .npy files.
    """
    X_test = np.load('data/test/X_test.npy')
    scaling_test = np.load('data/test/scaling_test.npy')
    ids_test = np.load('data/test/ids_test.npy')
    y_test = np.load('data/test/y_test.npy')

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(X_test)
    np.random.seed(seed)
    np.random.shuffle(scaling_test)
    np.random.seed(seed)
    np.random.shuffle(ids_test)
    np.random.seed(seed)
    np.random.shuffle(y_test)

    return X_test, scaling_test, ids_test, y_test

def split_data(X, scaling, ids, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param scaling: pixel spacing
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = int(X.shape[0] * split_ratio) # index must be int
    X_test = X[:split, :, :, :]
    scaling_test = scaling[:split, :]
    ids_test = ids[:split]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    scaling_train = scaling[split:, :]
    ids_train = y[split:]
    y_train = y[split:, :]

    return X_train, scaling_train, ids_train, y_train, X_test, scaling_test, ids_test, y_test

def train():
    """
    Training systole and diastole models.
    """
    print('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    # load the preprocessed data with the heart cut-out
    print('Loading data...')
    X_train, scaling_train, ids_train, y_train = load_train_data()
    X_test, scaling_test, ids_test, y_test = load_test_data()

    nb_iter = 200       # a higher number seems to give rise to overfitting
    epochs_per_iter = 3 # reduces overfitting
    batch_size = 32     # not tuned - potential improvement
    calc_crps = 2       # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        # augment data to make up for low number of samples
        print('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, 15)
        print('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)

        print('Fitting systole model...')
        hist_systole = model_systole.fit([X_train_aug, scaling_train], y_train[:, 0], shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data=([X_test, scaling_test], y_test[:, 0]))

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit([X_train_aug, scaling_train], y_train[:, 1], shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data=([X_test, scaling_test], y_test[:, 1]))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            print('Evaluating CRPS...')
            pred_systole = model_systole.predict([X_train, scaling_train], batch_size=batch_size, verbose=1)
            pred_diastole = model_diastole.predict([X_train, scaling_train], batch_size=batch_size, verbose=1)
            val_pred_systole = model_systole.predict([X_test, scaling_test], batch_size=batch_size, verbose=1)
            val_pred_diastole = model_diastole.predict([X_test, scaling_test], batch_size=batch_size, verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
            cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

            # CDF for predicted data
            cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            print('CRPS(test) = {0}'.format(crps_test))

        print('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights('weights_systole.hdf5', overwrite=True)
        model_diastole.save_weights('weights_diastole.hdf5', overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights('weights_systole_best.hdf5', overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights('weights_diastole_best.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('val_loss.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))

if __name__=="__main__":
    train()
