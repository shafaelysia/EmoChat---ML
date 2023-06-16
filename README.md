# EmoChat - Machine Learning
## Speech Emotion Recognition Model
EmoChat incorporates advanced machine learning techniques to enhance the user experience and enable emotion recognition in voice notes. By leveraging state-of-the-art algorithms and deep learning models, EmoChat can identify and analyze human emotions embedded in audio. EmoChat utilizes a machine learning model trained using TensorFlow, an open-source machine learning framework. The model is a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units.

## Tools and Library
* Jupyter Notebook
* Tensorflow
* Librosa
* Numpy
* scikit-learn
* json

## Datasets
1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):

Dataset Link: [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
Description: RAVDESS contains high-quality audio recordings of actors portraying various emotions, including neutral, happy, sad, fear, angry, surprise, and disgust. The dataset offers a wide range of emotional expressions and vocal characteristics for training the model.

2. CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset):

Dataset Link: [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)
Description: CREMA-D comprises audio and visual recordings of actors performing scripted scenarios with emotional expressions. The dataset includes a diverse set of emotional states, allowing the model to learn and recognize emotions such as neutral, happy, sad, anger, and more.

3. SAVEE (Surrey Audio-Visual Expressed Emotion):

Dataset Link: [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
Description: SAVEE contains speech audio recordings from four male actors expressing seven different emotions, including neutral, happiness, sadness, anger, fear, disgust, and surprise. The dataset provides a valuable resource for training the model to identify and differentiate various emotional states.

4. TESS (Toronto Emotional Speech Set):

Dataset Link: [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
Description: TESS comprises audio recordings of professionally trained actors simulating emotional expressions in sentences. The dataset covers emotions such as anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral. TESS contributes to training the model to accurately recognize and classify these emotional states.

## Feature Extraction
To analyze and recognize emotions in voice notes, EmoChat employs feature extraction techniques to capture essential characteristics from the audio data. One of the prominent features used in this process is Mel-frequency cepstral coefficients (MFCCs), which are extracted using the 'librosa' library in Python.
MFCCs are a representation of the short-term power spectrum of an audio signal, capturing both the spectral and temporal aspects of the voice notes.

## Emotion Categories
The trained model is capable of recognizing six distinct emotions: neutral, happy, sad, fear, angry, and disgust. This comprehensive set of emotion categories allows EmoChat to identify and analyze the emotional content of voice notes.

## Model Archiecture
The emotion recognition model in EmoChat is built using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), specifically LSTM. This architecture allows the model to effectively extract features from the MFCC inputs and make predictions for different emotions.

The model consists of multiple layers, including convolutional layers, recurrent layers, and dense layers. Each layer plays a crucial role in processing the input data and capturing relevant patterns. The model architecture is designed to learn and recognize emotional cues present in the voice notes.

Below is a screenshot of the model summary, providing an overview of the layer configurations and parameter counts:
![Model Summary]()

## Training and Evaluation
During the training phase, the model undergoes an iterative process that involves forward propagation, backpropagation, and gradient descent optimization. This process aims to minimize the difference between predicted emotions and the ground truth labels in the training datasets. The model adjusts its internal parameters to learn the underlying relationships between the extracted audio features and the corresponding emotions.

Following the training process, the model's performance is evaluated on a separate test set to assess its accuracy and effectiveness. EmoChat's emotion recognition model demonstrates impressive accuracy, enabling users to better understand the emotional context of the voice notes they receive.

## Performance Analysis
The performance analysis of EmoChat's emotion recognition model is a crucial aspect of its development. While the model has shown promising results, it is important to acknowledge the ongoing challenge of improving accuracy. Currently, the accuracy on the test set has peaked at 67%, indicating room for further enhancements.

Several factors contribute to the complexity of accurately recognizing emotions in voice notes. These include variations in speech patterns, individual vocal characteristics, and the subjective nature of emotions themselves. Despite efforts to leverage advanced machine learning techniques and feature extraction with MFCCs, achieving higher accuracy remains an active area of research and development.

The EmoChat team continues to explore various strategies to enhance model performance. This includes fine-tuning the model architecture, increasing the diversity and size of the training dataset, and implementing techniques such as data augmentation and transfer learning. These approaches aim to capture a wider range of emotional nuances and improve the model's ability to generalize to unseen data.
