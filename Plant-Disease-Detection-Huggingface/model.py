import gradio as gr
import tensorflow as tf

from tensorflow_addons.optimizers import CyclicalLearningRate
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Add, SeparableConv2D
from keras.models import Model

@tf.autograph.experimental.do_not_convert
class ResnetBlock(Model):
    """
    A standard resnet block.
    """
    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = SeparableConv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.swish(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        x = self.merge([x, res])
        out = tf.nn.swish(x)
        return out

@tf.autograph.experimental.do_not_convert
class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (3, 3), strides=2, padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

model = ResNet18(38)
model.build(input_shape = (None,256,256,3))
cyclical_learning_rate = CyclicalLearningRate(
    initial_learning_rate=3e-7,
    maximal_learning_rate=0.001,
    step_size=38,
    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    scale_mode='cycle'
)

optimizer = tf.keras.optimizers.Adam(learning_rate = cyclical_learning_rate, clipvalue=0.1)                         
model.compile(loss="categorical_crossentropy", optimizer =optimizer, metrics=["accuracy"])
model.load_weights('model_weights.x5')

labels = {
    0: 'Apple Scab',
    1: 'Apple Black Rot',
    2: 'Cedar Apple Rust',
    3: 'Healthy Apple',
    4: 'Healthy Blueberry',
    5: 'Healthy Cherry',
    6: 'Cherry Powdery Mildew',
    7: 'Corn Gray Leaf Spot',
    8: 'Corn Common Rust',
    9: 'Corn Northern Leaf Blight',
    10: 'Healthy Corn',
    11: 'Grape Black Rot',
    12: 'Grape Esca (Black Measles)',
    13: 'Grape Leaf Blight (Isariopsis Leaf Spot)',
    14: 'Healthy Grape',
    15: 'Orange Haunglongbing (Citrus Greening)',
    16: 'Peach Bacterial Spot',
    17: 'Healthy Peach',
    18: 'Bell Pepper Bacterial Spot',
    19: 'Healthy Bell Pepper',
    20: 'Potato Early Blight',
    21: 'Potato Late Blight',
    22: 'Healthy Potato',
    23: 'Healthy Raspberry',
    24: 'Healthy Soybean',
    25: 'Squash Powdery Mildew',
    26: 'Strawberry Leaf Scorch',
    27: 'Healthy Strawberry',
    28: 'Tomato Bacterial Spot',
    29: 'Tomato Early Blight',
    30: 'Tomato Late Blight',
    31: 'Tomato Leaf Mold',
    32: 'Tomato Septoria Leaf Spot',
    33: 'Tomato Spider Mites',
    34: 'Tomato Target Spot',
    35: 'Tomato Yellow Leaf Curl Virus',
    36: 'Tomato Mosaic Virus',
    37: 'Healthy Tomato'
}
imgSize = 200
def classify_image(inp):
    inp = inp.reshape(-1, imgSize, imgSize, 3)
    inp = tf.cast(inp, tf.float32)
    prediction = model.predict(inp)
    return {f'{i}: {labels[i]}': float(prediction[0][i]) for i in range(len(labels)-1)}

image = gr.inputs.Image(shape=(imgSize, imgSize))
label = gr.outputs.Label(num_top_classes=1)

gr.Interface(fn=classify_image, inputs=image, outputs=label, capture_session=True).launch()