"""
Usage:
    model.py --train_data=<train_data> --test_data=<test_data> --audio_feature=<audio_feature> [--embedder_checkpoint=<checkpoint_dir> --num_epochs=<num_epochs>] 

Options:
  --train_data=<train_data> .............. a csv file with columns audio_file_path,text,tag
  --test_data=<test_data> ................ a csv file with columns audio_file_path,text,tag
  --audio_feature=<audio_feature> ....... audio feature to be extracted available options are mel_spectogram,opensmile
  --embedder_checkpoint=<checkpoint_dir>  ................. the sentence embedding checkpoint directory deafult is roberta-large-nli-stsb-mean-tokens 
  --num_epochs=<num_epochs>......... the number of epochs to train
"""


import os
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout,Input,Concatenate
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
import numpy as np
import pandas as pd
import random
import soundfile
import librosa
import pandas as pd 
from tqdm import tqdm
import glob
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import classification_report
from pprint import pprint
from docopt import docopt


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


import torch


def get_model(text_feature_shape,acoustic_feature_shape,n_classes=2,acoustic_sequence=False):
    x_in_emb = Input(shape=text_feature_shape)
    x_in_spec = Input(shape=acoustic_feature_shape)
    if acoustic_sequence:
        x_rnn = GRU(128)(x_in_spec)
    else:
        x_rnn=x_in_spec
    x_merged = Concatenate()([x_rnn, x_in_emb])
    x_clf_2 = Dense(64, activation='relu')(x_merged)
    if n_classes==2:
        x_clf_1 = Dense(1, activation='sigmoid')(x_clf_2)
        model=Model(inputs=[x_in_emb,x_in_spec],outputs=x_clf_1)
        model.compile(loss='binary_crossentropy', metrics=["accuracy"], optimizer='adam')
    else:
        x_clf_1 = Dense(n_classes, activation='softmax')(x_clf_2)
        model=Model(inputs=[x_in_emb,x_in_spec],outputs=x_clf_1)
        model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer='adam')
    return model     


def extract_mel_spectogram(file_name, **kwargs):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        mel_spec=librosa.feature.melspectrogram(X, sr=sample_rate).T
    return mel_spec


def get_feature_list(data_frame,audio_featurizer="mel_spectogram",text_featurizer="sentence_embeddings",checkpoint="roberta-large-nli-stsb-mean-tokens"):
    X_acoustic=[]
    Y=[]
    texts=[]
    if audio_featurizer=="opensmile":
        import opensmile
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        )
    for i in range(len(data_frame)):
        if audio_featurizer=="mel_spectogram":
            raw_mel=extract_mel_spectogram(data_frame.iloc[i]['audio_file_path'],mel=True)
            X_acoustic.append(raw_mel)
        elif audio_featurizer=="opensmile":
            y = smile.process_file(model_train_DataFrame.iloc[i]['audio_file_path'])
            X_acoustic.append([y.iloc[0][col] for col in y.columns])
            
        Y.append(data_frame.iloc[i]['tag'])
        texts.append(data_frame.iloc[i]['text'])
    if text_featurizer=="sentence_embeddings":
        from sentence_transformers import SentenceTransformer
        if checkpoint is None:
            embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
        else:
            embedder = SentenceTransformer(checkpoint)
        X_textual= embedder.encode(texts)
    X_acoustic_padded=pad_sequences(X_acoustic,padding='post')
    le=LabelEncoder()
    Y=le.fit_transform(Y)
    if len(X_acoustic_padded[0].shape)>1:
        X_acoustic_shape=(None,X_acoustic_padded[0].shape[1])
    else:
        X_acoustic_shape=(X_acoustic_padded[0].shape[0])
    print(X_acoustic_shape)    
    if len(X_textual[0].shape)>1:
        X_textual_shape=(None,X_textual[0].shape)
    else:
        X_textual_shape=(X_textual[0].shape[0])    
    if (audio_featurizer=="mel_spectogram") and text_featurizer=="sentence_embeddings":
        model=get_model(X_textual_shape,X_acoustic_shape,n_classes=len(le.classes_),acoustic_sequence=True)
    elif audio_featurizer=="opensmile" and text_featurizer=="sentence_embeddings": 
        model=get_model(X_textual_shape,X_acoustic_shape,n_classes=len(le.classes_),acoustic_sequence=False)
        
    return [np.array(X_textual),np.array(X_acoustic_padded)],np.array(Y),le,model



def main():
    args        = docopt(__doc__)
    train_csv = args.get("--train_data")
    test_csv = args.get("--test_data")
    audio_feature = args.get("--audio_feature")
    checkpoint_dir = args.get("--embedder_checkpoint")
    epochs=args.get('--num_epochs')
    if epochs is not None:
        epochs=int(epochs)
    else:
        epochs=50    
    train_DataFrame=pd.read_csv(train_csv)
    test_DataFrame=pd.read_csv(test_csv)
    X,Y,labelencoder,model=get_feature_list(train_DataFrame.iloc[:20],audio_featurizer=audio_feature,text_featurizer="sentence_embeddings",checkpoint=checkpoint_dir)
    model.fit(X,Y,epochs=epochs,batch_size=8)
    X_test,Y_test,_,_=get_feature_list(test_DataFrame,audio_featurizer=audio_feature,text_featurizer="sentence_embeddings")
    model_prediction = model.predict(X_test)
    if len(labelencoder.classes_)==2:
        predicted_label=[]
        for i,predicted in enumerate(model_prediction):
            print(predicted[0])
            if predicted[0]>0.5:
                predicted_label.append(1)
            else:
                predicted_label.append(0)            
        pred_labels=labelencoder.inverse_transform(predicted_label)
    else:
        pred_labels=labelencoder.inverse_transform(np.argmax(model_prediction,axis=1))
    pprint(classification_report(test_DataFrame['tag'].to_list(),pred_labels))


if __name__=='__main__':
    main()





