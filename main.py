
from modeling_tools import ModelClass
from data_tools import create_train_test_val
from image_tools import augmentation_controller, resize_images
from glob import glob


if __name__ == '__main__':

    classes = ['bathroom', 'bedroom', 'exterior', 'kitchen', 'living room']

    # --- create train, test, and validation datasets from existing labeled data  --- #

    create_train_test_val(img_dir='./images/', split_dir='./custom-split/', weights=(0.85, 0.1, 0.05), total_files=1500)

    # --- resize images to (150, 150, 3) --- #

    resize_images(glob(r'./custom-split/*/*/*.jpg'))

    # --- create additional synthetic training data from the existing train data --- #

    augmentation_controller(iterations=5, train_dir=r'./custom-split/train/', classes=classes, rebuild=False)

    # --- instantiate the model class which will manage the data pipelines, training, and evaluation --- #

    model = ModelClass(b_size=50, log_dir=r'./logs', model_dir=r'./models', img_dir=r'./custom-split', classes=classes, img_size=(150, 150))

    # --- set up model storage and tensorboard log directories --- #

    model.create_model_dir()
    model.create_log_dir()

    # --- set up custom pipeline for managing very large data ingestion --- #

    model.custom_pipeline()

    # --- build and train model --- #

    model.build_model(epochs=50)

    # --- produce confusion matrix with the test data -- #

    model.get_test_results(model_path='models/rooms.model.best.hdf5')

