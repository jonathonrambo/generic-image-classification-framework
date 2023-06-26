
import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.metrics import confusion_matrix

import itertools

from glob import glob
from os import path, makedirs
from shutil import rmtree



class ModelClass:
    def __init__(self, b_size, log_dir, model_dir, classes, img_dir, img_size=(150, 150)):

        self.b_size = b_size
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.classes = classes
        self.img_size = img_size
        self.img_dir = img_dir
        self.model = None
        self.train_ds, self.test_ds, self.val_ds = None, None, None


    def plot_confusion_matrix(self, cm, title='Confusion matrix'):

        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)


        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    def get_test_results(self, model_path=None, test_dir=None):

        if model_path:
            model_tmp = load_model(model_path)
        else:
            model_tmp = self.model

        if self.test_ds:
            test = self.test_ds
        elif test_dir:
            test = tf.keras.utils.image_dataset_from_directory(test_dir, label_mode='categorical',
                                                               class_names=self.classes, color_mode='rgb',
                                                               validation_split=None, batch_size=self.b_size,
                                                               shuffle=True, image_size=self.img_size)
        else:
            raise Exception

        x_test = np.concatenate([x for x, _ in test], axis=0)
        y_test = [self.classes[i] for i in [y.numpy().argmax() for _, y in test.unbatch()]]

        y_pred = model_tmp.predict(x_test)
        y_pred = [self.classes[i] for i in y_pred.argmax(axis=1)]

        confusion_mtx = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(confusion_mtx)


    def configure_for_performance_custom(self, ds):
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        ds = ds.cache()
        ds = ds.batch(self.b_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        ds = self.normalize(ds)
        return ds


    def custom_pipeline_worker(self, mask):

        AUTOTUNE = tf.data.AUTOTUNE
        image_count = len(list(glob(mask)))

        data_tmp = tf.data.Dataset.list_files(mask, shuffle=False)
        data_tmp = data_tmp.shuffle(reshuffle_each_iteration=False, buffer_size=image_count)
        data_tmp = data_tmp.map(self.process_path, num_parallel_calls=AUTOTUNE)

        return self.configure_for_performance_custom(data_tmp)


    def custom_pipeline(self, val_ratio=0.2):

        self.train_ds = self.custom_pipeline_worker(f'{self.img_dir}/train/*/*.jpg')
        self.test_ds = self.custom_pipeline_worker(f'{self.img_dir}/test/*/*.jpg')
        self.val_ds = self.custom_pipeline_worker(f'{self.img_dir}/val/*/*.jpg')


    def get_label(self, file_path, class_names):

        parts = tf.strings.split(file_path, path.sep)
        one_hot = parts[-2] == class_names

        return tf.where(one_hot, 1, 0)


    def decode_img(self, img):
        img = tf.io.decode_jpeg(img, channels=3)
        return img


    def process_path(self, file_path):

        label = self.get_label(file_path, self.classes)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        return img, label


    def normalize(self, ds):
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        ds = ds.map(lambda x, y: (normalization_layer(x), y))
        return ds


    def create_log_dir(self):
        if path.exists(self.log_dir):
            rmtree(self.log_dir)
        makedirs(self.log_dir)


    def create_model_dir(self):
        if path.exists(self.model_dir):
            rmtree(self.model_dir)
        makedirs(self.model_dir)


    def build_model(self, epochs):

        self.model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

        for layer in self.model.layers:
            layer.trainable = False

        x = layers.Flatten()(self.model.output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(5, activation='sigmoid')(x)
        self.model = Model(self.model.input, x)

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, min_delta=0.01)
        checkpoint = ModelCheckpoint(filepath=r'./models/rooms.model.best.hdf5', monitor='val_loss', mode='min', save_best_only=True)
        tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)

        self.model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', 'categorical_crossentropy'])
        self.model.fit(self.train_ds, batch_size=self.b_size, validation_data=self.val_ds, epochs=epochs, callbacks=[early_stopping, checkpoint, tensorboard_callback])

        return self.model



