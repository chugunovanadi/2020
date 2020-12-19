import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K


def learn(base_weigths, model_name, epoch, base_epoch):
    json_file = open(model_name + "_arch.json", 'r')
    loaded_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_json)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    K.set_image_dim_ordering('th')
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]

    model.load_weights(base_weigths)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128,
              epochs=epoch, verbose=1, validation_split=0.2)

    score = model.evaluate(x_test, y_test, verbose=1)
    print("Test score: ", score[0])
    print("Test acc: ", score[1])

    epoch += base_epoch
    model.save_weights('weigths/' + model_name + '_' + str(epoch) + '_weig.h5')
    # 8: score: 0.08126533157732338  acc: 0.9741
    # 10: score: 0.06524057221990079 acc: 0.9795
    # 14: score: 0.05495657164007425 acc: 0.9817
    # 17: score: 0.0539248748447746 acc: 0.983
    # 20: score: 0.05988883427558467 acc: 0.9819


if __name__ == '__main__':
    base_weight = 'weigths/LeNet_10_weig.h5'
    model_name = 'LeNet'
    epoch = 3
    base_epoch = 17
    learn(base_weight, model_name, epoch, base_epoch)
