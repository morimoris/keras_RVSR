import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Concatenate, Lambda, Average

def ESPCN(input_list, input_channels, mag):
    if len(input_list) == 1:
        input_shape = input_list[0]
    else:
        input_shape = Concatenate()(input_list)

    conv2d_0 = Conv2D(filters = len(input_list) * input_channels,
                        kernel_size = (5, 5),
                        padding = "same",
                        activation = "relu",
                        )(input_shape)
    conv2d_1 = Conv2D(filters = 32,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu",
                        )(conv2d_0)
    conv2d_2 = Conv2D(filters = mag ** 2,
                        kernel_size = (3, 3),
                        padding = "same",
                        )(conv2d_1)


    pixel_shuffle = Lambda(lambda z: tf.nn.depth_to_space(z, mag))(conv2d_2)
        
    return pixel_shuffle

def RVSR(input_LR_num, input_channels, mag):
    input_list = input_LR_num * [None]
    output_list = (input_LR_num // 2 + 1) * [None]

    for img in range(input_LR_num): 
        input_list[img] = Input(shape = (None, None, input_channels), name = "input_" + str((img)))

    for num in range(0, input_LR_num // 2 + 1):
        output = ESPCN(input_list[input_LR_num // 2 - num : input_LR_num // 2 + num + 1], input_channels, mag)
        output_list[num] = output
    
    Tem_agg_model = Average()(output_list)

    model = Model(inputs = input_list, outputs = [Tem_agg_model])

    model.summary()
    return model
        





    
