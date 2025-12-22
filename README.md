### Models :
This folder contains notebooks of the model's training.
Each notebook aims to train a model and/or use it.
- audioLDM and Conditioned_LSTM contain code blocks to download, train, and uses their respective models

### Metrics : 
- Music_score contains the CNN used to do latent representation of spectrogram as well as code blocks to import the generated wav files and functions to compute their scores.

### Config files : 
- audioLDM requires a configuration file in the YAML extension for training and inference. In order for anyone to reproduce our results, we suggest using the given file.

### Utils :
- The file slice is just a script to cut a wav file into 30 seconds slices.
- labelling contains the code that uses the panns_inference python library to put labels on the music files. These labels are used by other models.
