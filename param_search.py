import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from pipeline import *

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, GRU, LSTM, Masking
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import binary_accuracy
from keras.regularizers import l2

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score


calls, orders = load_cleaned_data(
'dataset1_anon/cleaned/calls_cleaned.csv',
'dataset1_anon/cleaned/orders_cleaned.csv')


dates = [pd.to_datetime('2016-02-01 04:00:00') +
 pd.Timedelta(24*k,'D') for k in range(12)]

preprocess_params = [('makehist', MakeHistories()),
                     ('sampleandlabel', SampleAndLabel(
                         dates=dates,
                         churn_metric='calls',
                         churn_days=90,
                         min_calls=10,
                         min_paidorders=1,
                         lastcall_window=(90,0),
                         lastorder_window=(90,0),
                         allow_person_resample=False,
                         sample_end=None,
                         events_remaining=0)),
                     ('makefeats', MakeSequenceFeatures() ),
                     ('shufflesplit', ShuffleAndSplit(
                         test_frac=0.15,
                         return_ids=False) ),
                     ('makepadded', MakePaddedSequences(
                         return_ids=False,
                         maxlen=40))
                    ]

prepare_data = Pipeline(preprocess_params)

train_set, test_set, train_labels, test_labels = (
    prepare_data.transform( [calls, orders] ) )

# Validation set (for early stopping)
indices = np.arange(train_set.shape[0])
np.random.shuffle(indices)
val_frac = 0.15/0.85
split = round(val_frac * train_set.shape[0])

val_set = train_set[indices[:split], :, :]
train_set = train_set[indices[split:], :, :]
val_labels = train_labels.iloc[indices[:split]]
train_labels = train_labels.iloc[indices[split:]]

print('train_set shape: ', train_set.shape, 'test_set shape: ', test_set.shape)

print('Class ratio (train set): \n', train_labels.value_counts(normalize=True),
      '\n Class ratio (test set): \n', test_labels.value_counts(normalize=True))


param_grid = [{}, {}, ...]


def create_model(
    rnn_type='LSTM',
    dense_layers=1,
    rnn_layers=1,
    dense_size=16,
    rnn_size=32,
    epochs=10,
    lr=0.005,
    decay=0.0,
    earlystopping=False,
    batch_size=64,
    val_split=0.15,
    dropout=0.3,
    rec_dropout=0.3,
    dense_l2=1e-4,
    rec_kern_l2=1e-4,
    rec_l2=1e-4):

    assert rnn_type in ['GRU', 'LSTM']
    optim = Adam(lr=lr,
                 decay=decay)
    loss = 'binary_crossentropy'
    callbacks=None
    if earlystopping:
        early_stopping_monitor = EarlyStopping(
            monitor='f1_score', patience=20, restore_best_weights=True )
        callbacks=[early_stopping_monitor]
    num_features = train_set.shape[-1]
    max_len = train_set.shape[1]
    input_shape = (max_len, num_features)

    model = Sequential()
    model.add(
        Masking(mask_value=0., input_shape=(max_len, num_features)))

    for _ in range(dense_layers):
        model.add( TimeDistributed(
            Dense(dense_size, activation='tanh',
                  kernel_regularizer=l2(dense_l2) )))

    for layer in range(rnn_layers):
        if rnn_type == 'GRU':
            model.add(
                GRU(rnn_size, dropout=dropout,
                           recurrent_dropout=rec_dropout,
                           kernel_regularizer=l2(rec_kern_l2),
                           recurrent_regularizer=l2(rec_l2),
                           input_shape=(max_len, num_features),
                           return_sequences= (not (layer==(rnn_layers-1)))
                            ) )
        if rnn_type == 'LSTM':
            model.add(
                LSTM(
                    rnn_size, dropout=dropout,
                    recurrent_dropout=rec_dropout,
                    kernel_regularizer=l2(rec_kern_l2),
                    recurrent_regularizer=l2(rec_l2),
                    input_shape=(max_len, num_features),
                    return_sequences= (not (layer==(rnn_layers-1)))
                     ) )

    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(dense_l2)))

    model.compile(loss=loss, optimizer=optim, metrics=['accuracy'])

    return model

model = KerasClassifier(build_fn=create_model,
                        verbose=0,
                        batch_size=128,
                        epochs=50,
                        shuffle=True,
                        # validation_data=(val_set, (1*val_labels).values),
                        rnn_type='LSTM',
                        dense_layers=1,
                        rnn_layers=1,
                        dense_size=16,
                        rnn_size=32,
                        lr=0.005,
                        decay=0.0,
                        earlystopping=True,
                        val_split=0.15,
                        dropout=0.3,
                        rec_dropout=0.3,
                        dense_l2=0.001
                        )

# RandomizedSearchCV or GridSearchCV
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           verbose=10)
grid_search.fit(train_set, (1*train_labels).values)

print('Best parameters: ', '\n', grid_search.best_params_)

results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('grid_search_results.csv')
