## Introduction <br />
This repo has code base to create multi modal emotion classifiers using acoustic and textual features. 
This project aims to explore different types of acoustic features and textual features for modelling emotions in a conversational setting and also the make emperical process of comparing each technique faster   

### Audio features currently added are :
1. Mel spectogram using librosa 
2. Opensmile's ComParE_2016 setting (https://www.audeering.com/opensmile/)

### Textual features currently added are: 
1. Sentence-bert https://www.sbert.net/ you can see the default model or use a custom checkpoint as well

### Installation <br />

poetry install 

### Usage <br />

  model.py --train_data=<train_data> --test_data=<test_data> --audio_feature=<audio_feature> [--embedder_checkpoint=<checkpoint_dir> --num_epochs=<num_epochs>] 

### Options <br />
  --train_data=<train_data> .............. a csv file with columns audio_file_path,text,tag <br />
  --test_data=<test_data> ................ a csv file with columns audio_file_path,text,tag <br />
  --audio_feature=<audio_feature> ....... audio feature to be extracted available options are mel_spectogram,opensmile <br />
  --embedder_checkpoint=<checkpoint_dir>  ....... the sentence-bert checkpoint deafult is roberta-large-nli-stsb-mean-tokens <br />  
  --num_epochs=<num_epochs>......... the number of epochs to train 


