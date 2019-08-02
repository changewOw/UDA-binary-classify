from keras.applications.densenet import DenseNet121
import keras.layers as kl
import keras.models as km
import keras.backend as K
import tensorflow as tf

def labeled_data_loss(y_true, y_pred):
    # y_true: (B,H,W)
    # y_pred: (B,H,W,2)
    return K.sparse_categorical_crossentropy(y_true, y_pred)

def unlabeled_data_loss(y_true, y_pred):
    """
    :param y_true: (1,) unused
    :param y_pred: (B,H,W,2) unlabeled data(sub)
    :return:
    """
    x = tf.nn.l2_loss(y_pred)
    x = K.mean(x)
    return x


def classified(classify_name, input_shape):
    if classify_name == 'densenet121':
        classify = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise NotImplementedError
    x = classify.output
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.5)(x)
    x = kl.Dense(512, activation='relu')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Dropout(0.5)(x)
    x = kl.Dense(2)(x) # 二分类问题  这里不加softmax激活
    model = km.Model(inputs=[classify.input], outputs=[x])
    return model

def softmaxT_func(T=1.):
    def softmax_T(x):
        # x is logits (B,H,W,2)
        x = x / K.constant(T, dtype=tf.float32)
        x = K.softmax(x)
        return x
    return softmax_T


def UDA_model(input_size=(256, 256, 3), T=1.):

    labeled_data = kl.Input(shape=input_size, name='labeled_data_input')
    unlabeled_data = kl.Input(shape=input_size, name='unlabeled_data_input')
    aug_unlabeled_data = kl.Input(shape=input_size, name='aug_unlabeled_data_input')

    classify = classified('densenet121', input_shape=input_size)

    # 有标签的data
    labeled_data_out = classify(labeled_data)
    unlabeled_data_out = classify(unlabeled_data)
    aug_unlabeled_data_out = classify(aug_unlabeled_data)
    unlabeled_data_out = kl.Lambda(lambda x:K.stop_gradient(x))(unlabeled_data_out) # according to paper, not propagated back
    unlabeled_data_out = kl.Lambda(softmaxT_func(T))(unlabeled_data_out) # softmax with temperature T


    labeled_data_out = kl.Activation('softmax', name='labeled_data_out')(labeled_data_out)
    unlabeled_data_subout = kl.Subtract(name='unlabeled_data_sub_out')([unlabeled_data_out, aug_unlabeled_data_out])


    model = km.Model(inputs=[labeled_data, unlabeled_data, aug_unlabeled_data],
                     outputs=[labeled_data_out, unlabeled_data_subout])
    return model

if __name__ == '__main__':
    train_size = 256
    loss = {
        'labeled_data_out':labeled_data_loss,
        'unlabeled_data_sub_out':unlabeled_data_loss
    }
    loss_weights = {
        'labeled_data_out':1.,
        'unlabeled_data_sub_out':20. # according to paper 20 imagenet cofig
    }
    target_tensor = {
        'unlabeled_data_sub_out':K.placeholder(shape=(1,), dtype=tf.uint8)
    }

    model = UDA_model(input_size=(train_size, train_size, 3), T=0.4) # according to paper 0.4 imagenet config
    model.compile(loss=loss, loss_weights=loss_weights, optimizer='adam', target_tensors=target_tensor)
    model.summary()









