# ML dependencies
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

# This defines the 1-stack U-Net architecture. 
def get_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Creates the standard U-Net model. 

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(1, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = c9

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# This defines the 2-stack U-Net architecture. 
def get_2stack_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Creates the 2-stack U-net model. 

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # --------------------------------------------------------------------
    # U-Net 1:

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)


    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    # --------------------------------------------------------------------

    # U-Net 2: 

    c12 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    c12 = Dropout(0.1) (c12)
    c12 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c12)
    p12 = MaxPooling2D((2, 2)) (c12)

    c22 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p12)
    c22 = Dropout(0.1) (c22)
    c22 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c22)
    p22 = MaxPooling2D((2, 2)) (c22)

    c32 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p22)
    c32 = Dropout(0.2) (c32)
    c32 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c32)
    p32 = MaxPooling2D((2, 2)) (c32)

    c42 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p32)
    c42 = Dropout(0.2) (c42)
    c42 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c42)
    p42 = MaxPooling2D(pool_size=(2, 2)) (c42)

    c52 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p42)
    c52 = Dropout(0.3) (c52)
    c52 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c52)

    u62 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c52)
    u62 = concatenate([u62, c42])
    c62 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u62)
    c62 = Dropout(0.2) (c62)
    c62 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c62)

    u72 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c62)
    u72 = concatenate([u72, c32])
    c72 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u72)
    c72 = Dropout(0.2) (c72)
    c72 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c72)

    u82 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c72)
    u82 = concatenate([u82, c22])
    c82 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u82)
    c82 = Dropout(0.1) (c82)
    c82 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c82)


    u92 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c82)
    u92 = concatenate([u92, c12], axis=3)
    c92 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u92)
    c92 = Dropout(0.1) (c92)
    c92 = Conv2D(1, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c92)

    # --------------------------------------------------------------------
    outputs = c92

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# This defines a deeper 2-stack U-Net architecture. 
def get_2stack_unet_deeper(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Creates the 2-stack U-net model, but with some additional conv layers at lower levels. 

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # --------------------------------------------------------------------
    # U-Net 1:

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    # added depth here
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)    
    p2 = MaxPooling2D((2, 2)) (c2)

    # added depth here
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    # added depth here
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    # added depth here
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    # added depth here
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    # added depth here
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)


    # --------------------------------------------------------------------
    # U-Net 2: 

    c12 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    c12 = Dropout(0.1) (c12)
    c12 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c12)
    p12 = MaxPooling2D((2, 2)) (c12)

    # added depth here
    c22 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p12)
    c22 = Dropout(0.1) (c22)
    c22 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c22)
    c22 = Dropout(0.1) (c22)
    c22 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c22)
    p22 = MaxPooling2D((2, 2)) (c22)

    # added depth here
    c32 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p22)
    c32 = Dropout(0.2) (c32)
    c32 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c32)
    c32 = Dropout(0.2) (c32)
    c32 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c32)
    p32 = MaxPooling2D((2, 2)) (c32)

    # added depth here
    c42 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p32)
    c42 = Dropout(0.2) (c42)
    c42 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c42)
    c42 = Dropout(0.2) (c42)
    c42 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c42)
    p42 = MaxPooling2D(pool_size=(2, 2)) (c42)

    c52 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p42)
    c52 = Dropout(0.3) (c52)
    c52 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c52)

    # added depth here
    u62 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c52)
    u62 = concatenate([u62, c42])
    c62 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u62)
    c62 = Dropout(0.2) (c62)
    c62 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c62)
    c62 = Dropout(0.2) (c62)
    c62 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c62)

    # added depth here
    u72 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c62)
    u72 = concatenate([u72, c32])
    c72 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u72)
    c72 = Dropout(0.2) (c72)
    c72 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c72)
    c72 = Dropout(0.2) (c72)
    c72 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c72)

    # added depth here
    u82 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c72)
    u82 = concatenate([u82, c22])
    c82 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u82)
    c82 = Dropout(0.1) (c82)
    c82 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c82)
    c82 = Dropout(0.1) (c82)
    c82 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c82)


    u92 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c82)
    u92 = concatenate([u92, c12], axis=3)
    c92 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u92)
    c92 = Dropout(0.1) (c92)
    c92 = Conv2D(1, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c92)

    # --------------------------------------------------------------------
    outputs = c92

    model = Model(inputs=[inputs], outputs=[outputs])
    return model 

if __name__ == '__main__':
    # example usage 
    model = get_2stack_unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
