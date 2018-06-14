import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib

import config
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument('--accuracy', action='store_true', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16])
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=500, type=int, help='How many files to predict on at once')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):#Judge whether "path" is a directory, return a boolean value
        files = glob.glob(path + '*.jpg')#return a list which contains file paths 
    elif path.find('*') > 0:#find a str in "path" and reture its location
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        x = model_module.load_img(i)#get picture whose type is ndarray, 3D
        try:
            image_class = i.split(os.sep)[-2]#chose separator automaticly in defferent system, store the directory which include pictures
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])#divide path to directory and filename, this is a tuple including two elements

        inputs.append(x)
        
    return y_true, inputs#so y_ture is the filename when executing excepy module, and inputs is ndarray


def predict(path):
    files = get_files(path)#get files' path list
    n_files = len(files)#get the number of picture
    print('Found {} files'.format(n_files))#print information

    if args.novelty_detection:#do not execute
        activation_function = util.get_activation_function(model, model_module.noveltyDetectionLayerName)
        novelty_detection_clf = joblib.load(config.get_novelty_detection_model_path())

    y_trues = []
    predictions = np.zeros(shape=(n_files,))#creat a matrix: n_files*1
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))#ceil, count the number of batch
    for n in range(0, nb_batch):
        print('Batch {}'.format(n))#print informatin
        n_from = n * args.batch_size#the number of beginning in current batch
        n_to = min(args.batch_size * (n + 1), n_files)#the number of end in current batch

        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        y_trues += y_true

        if args.store_activations:#do not execute
            util.save_activations(model, inputs, files[n_from:n_to], model_module.noveltyDetectionLayerName, n)

        if args.novelty_detection:#do not execute
            activations = util.get_activations(activation_function, [inputs[0]])
            nd_preds = novelty_detection_clf.predict(activations)[0]
            print(novelty_detection_clf.__classes[nd_preds])

        if not args.store_activations:
            # Warm up the model
            if n == 0:
                print('Warming up the model')
                start = time.clock()
                model.predict(np.array([inputs[0]]))
                end = time.clock()
                print('Warming up took {} s'.format(end - start))

            # Make predictions
            start = time.clock()
            out = model.predict(np.array(inputs))#predict!
            end = time.clock()
            predictions[n_from:n_to] = np.argmax(out, axis=1)#return the index of the maximum value of specified dimention
            print('Prediction on batch {} took: {}'.format(n, end - start))#print execution time

    if not args.store_activations:
        for i, p in enumerate(predictions):
            recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
            print('| should be {} ({}) -> predicted as {} ({})'.format(y_trues[i], files[i].split(os.sep)[-2], p,
                                                                       recognized_class))

        if args.accuracy:#do not execute
            print('Accuracy {}'.format(accuracy_score(y_true=y_trues, y_pred=predictions)))

        if args.plot_confusion_matrix:#do not execute
            cnf_matrix = confusion_matrix(y_trues, predictions)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=False)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=True)


if __name__ == '__main__':
    tic = time.clock()#return the current cpu time

    args = parse_args()
    print('=' * 50)
    print('Called with args:')
    print(args)

    if args.data_dir:#user defined directory
        config.data_dir = args.data_dir#~
        config.set_paths()#~
    if args.model:#user defined model
        config.model = args.model#~

    util.set_img_format()#channel_first or channels_last
    model_module = util.get_model_class_instance()#class model.resnet50
    model = model_module.load()#creat base_model and load trained weight!(ResNet50) "G:\keras-transfer-learning-for-oxford102\trained\fine-tuned-resnet50-weights.h5"

    classes_in_keras_format = util.get_classes_in_keras_format()#get a dictory of classes

    predict(args.path)#we must input a path of directory including pictures

    if args.execution_time:#print execution_time
        toc = time.clock()#~
        print('Time: %s' % (toc - tic))#~
