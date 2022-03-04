# Speech Recognition Using Deep Learning
This was my project for the Machine Learning course, during my Master, and it consisted of using deep learning for speech recognition. More specifically, recognizing which word is being played on an audio track.

I tried the experiment using the two main audio features: spectrograms and MFCCs (Mel Frequency Cepstral Coefficients). To run the implementation, first download the dataset (more instructions in the _dataset_ folder) and run one of the ``prepare_dataset.py`` files, depending on which feature you want to use. This ``python`` script will create a file called ``data.json`` in which there are the features that will be used to train the model. Then run the corresponding ``train.py`` file to train the model. When it has finished, the model will be saved (I provide two models already trained, ``model_spectograms.h5`` and ``model_mfccs.h5``). Finally, put in the test folder the tracks you want to make predictions about and run the corresponding ``predictions.py`` file.

Here I show the loss and accuracy curves I got when I did the project.

![Spectrogram_results](https://user-images.githubusercontent.com/71872419/156835006-6f6df845-3709-42d3-aa5c-99839120599c.jpg)

                                          Curves using sprectrograms

![MFCC_results](https://user-images.githubusercontent.com/71872419/156835068-6cf679e6-b684-4a08-91ed-36bea6d2d1d2.jpg)

                                             Curves using MFCCs
