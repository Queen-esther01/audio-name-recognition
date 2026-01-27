import os
import math
import glob
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.feature
import plotly.io as pio
# import sounddevice as sd
from pathlib import Path
import plotly.express as px
from sklearn import metrics
pio.renderers.default='browser'
import matplotlib.image as mpimg
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
# from keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from moviepy import VideoFileClip, AudioClip, concatenate_audioclips
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D


class Recogniser:
    def __init__(self, videos_path, audios_path):
        self.videos_path = videos_path
        self.audios_path = audios_path
        self.classes = ['ahmed', 'amber', 'charlie', 'christopher', 'dominic', 'emad', 'emma', 'hannah', 'imogen', 'jess',
                   'josh', 'joshua', 'kailong', 'kira', 'manwel', 'mateusz', 'ngozi', 'riley', 'sivaprasath', 'zack']
        self.epochs = 30
        self.batch_size = 32

    @staticmethod
    def get_classes():
        return ['ahmed', 'amber', 'charlie', 'christopher', 'dominic', 'emad', 'emma', 'hannah', 'imogen', 'jess',
                   'josh', 'joshua', 'kailong', 'kira', 'manwel', 'mateusz', 'ngozi', 'riley', 'sivaprasath', 'zack']

    def convert_video_to_audio_moviepy(self, video_file):
        """Converts video to audio using MoviePy library that uses `ffmpeg` under the hood"""
        # Get file names for labels
        stemFilename = (Path(os.path.basename(video_file)).stem)
        label = stemFilename.split('_')
        labels = label[0]

        if not os.path.exists('audio_files'):
            os.makedirs('audio_files')

        clip = VideoFileClip(video_file)
        fs = clip.fps
        clip.audio.write_audiofile(f"audio_files/{stemFilename}.wav")
        return fs, labels

    @staticmethod
    def extract_features(audio, max_frames):
        audio_mono, fs = librosa.load(audio, mono=True)
        mfcc = librosa.feature.mfcc(y=audio_mono, sr=fs, n_mfcc=40)  # returns shape (n_mfcc, n_frames)
        # UNCOMMENT TO CALCULATE MAX FRAMES IN THE MFCCS
        # max_frames = max(max_frames, mfcc.shape[1])
        # print(f'mfcc before padding; shape: {mfcc.shape}, ndim: {mfcc.ndim}, mfcc len: {len(mfcc)}, fs: {fs}')

        # Pad mfcc with 0 to meet up to max frame.
        mfcc_data = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])))

        # print(f'mfcc after padding; shape: {mfcc_data.shape}, ndim: {mfcc_data.ndim}, mfcc len: {len(mfcc_data)}')
        # print(f'max frames: {max_frames} \n')
        return mfcc_data

    def extract_audio_features(self):
        labels_data = []
        max_frames = 100
        data = []
        for audio in sorted(glob.glob(self.audios_path)):
            stemFilename = (Path(os.path.basename(audio)).stem)
            label = stemFilename.split('_')
            labels = label[0]
            labels_data.append(labels)

            mfcc = self.extract_features(audio, max_frames)
            data.append(mfcc)

        return data, labels_data, max_frames

    def one_hot_encoding(self, labels):
        LE = LabelEncoder()
        LE = LE.fit(self.classes)
        labels = to_categorical(LE.transform(labels))
        return labels, LE

    def create_model(self):
        numClasses = len(self.classes)
        model = Sequential()
        model.add(InputLayer(shape=(40, 100, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))
        return model

    def save_image(self, path, image):
        if not os.path.exists('report'):
            os.makedirs('report')
        mpimg.imsave(path, image)
        # cv.imwrite(path, image)

    def run(self):
        labels_data = []
        # data = []

        # # Step 1 - Extract audio from video
        # for file in sorted(glob.glob(self.videos_path)):
        #     fs, labels = self.convert_video_to_audio_moviepy(file)
        #     labels_data.append(labels)

        # Step 2 - Extract MFCC features from audio
        data, labels, max_frames = self.extract_audio_features()

        # Step 3 - Convert to numpy for NN
        labels = np.array(labels)
        data = np.array(data)
        print(f"label type: {type(labels)}, data type: {type(data)}")
        print(f"label shape: {labels.shape}, data shape: {data.shape}")

        # Step 4 - Normalise data so the input is in the range of 0 - 1 or -1 to 1
        data = data / np.max(data)

        # STEP 5 - One Hot Encoding
        labels, LE = self.one_hot_encoding(labels)

        # Step 6 - Split data
        # X_train is training data, y_train is are labels for training data
        # X_tmp is testing data, y_tmp is are labels for testing data
        X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

        # Step 7 - Train Model
        num_epochs = self.epochs
        num_batch_size = self.batch_size
        model = self.create_model()

        # specify optimizer & loss function
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size,  epochs=num_epochs, verbose=1)
        print(f'History: {history}')

        # # model.save_weights('audio.weights.h5')
        model.save("audio_model.keras")
        model.summary()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        # plt.savefig('accuracy.png')
        # plt.show()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        # plt.savefig('loss.png')
        # plt.show()

        # get accuracy
        predicted_probs = model.predict(X_test, verbose=0)
        predicted = np.argmax(predicted_probs, axis=1)
        actual = np.argmax(y_test, axis=1)
        accuracy = metrics.accuracy_score(actual, predicted)
        print(f'Accuracy: {accuracy * 100}%')

        # classification
        predicted_prob = model.predict(np.expand_dims(X_test[0, :, :], axis=0), verbose=0)
        predicted_id = np.argmax(predicted_prob, axis=1)
        predicted_class = LE.inverse_transform(predicted_id)
        print(f'predicted class: {predicted_class}')

        # confusion matrix
        confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot()
        # plt.savefig('confusion_matrix.png')
        # plt.show()


if __name__ == "__main__":
    videos_path = "/Users/star/Documents/MSC-LABS/audio-visual-processing-coursework-2/audio_video_files_v2/*.mov"
    audios_path = "audio_files/*.wav"
    recogniser = Recogniser(videos_path, audios_path)
    recogniser.run()