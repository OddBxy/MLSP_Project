### Model use :
This folder contains notebooks of the models training.
Each notebook aims to train a model and/or use it.
- Music_score contains the CNN use to do latent representation of spectrogram as well as code blocks to import the generated wav files and functions to compute their scores.
- labelling contains the code that use the panns_inference python library to put label on the music files. This label are used by other models.
- audioLDM and Conditioned_LSTM contains code blocks to download, train, and uses their respective models

### Config files : 
- audioLDM requires a configuration file in the yaml extension for training and inference. In order for anyone to reproduce our results, we suggest to use the given file.

### Utils :

