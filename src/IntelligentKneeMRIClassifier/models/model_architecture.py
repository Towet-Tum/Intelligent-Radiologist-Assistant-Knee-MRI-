from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def Knee_mri_3D_model(depth, height, width, channels, num_classes):
    input_layer = Input(shape=(depth, height, width, channels))

    # Convolutional layers
    conv1 = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv1 = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, kernel_size=(3, 3, 3), activation='relu')(pool1)
    conv2 = Conv3D(128, kernel_size=(3, 3, 3), activation='relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, kernel_size=(3, 3, 3), activation='relu')(pool2)
    conv3 = Conv3D(256, kernel_size=(3, 3, 3), activation='relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(512, kernel_size=(3, 3, 3), activation='relu')(pool3)
    conv4 = Conv3D(512, kernel_size=(3, 3, 3), activation='relu')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    flatten = Flatten()(pool3)

    dense1 = Dense(1024, activation='relu')(flatten)
    dense2 = Dense(1024, activation='relu')(dense1)
    dropout = Dropout(0.2)(dense2)

    output_layer = Dense(num_classes, activation='softmax')(dropout)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


