import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

# global
# Weights of the different loss components
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# using vgg for preprocessing
def preprocess_image(image_path):
    "Util function to open, resize and format pictures into appropriate tensors"
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols) )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # convert into vgg19 format`a
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

# reverse the preprocessing of vgg19
def deprocess_image(x):
    "Util function to convert a tensor into a valid image"
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# ------------------------------------------------------------
# loss functions
# ------------------------------------------------------------
# The gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    "compute image gram matrix"
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    "loss metric based on gram matrix"
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    # content is more global and abstract to be captured in the higher layers
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(x):
    "regularization loss"
    a = tf.square(        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]    )
    b = tf.square(        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# ------------------------------------------------------------
def compute_loss(feature_extractor, content_layer_name, content_weight, 
                style_layer_names, combination_image, base_image, style_reference_image):
    "find content loss and style loss"
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    #
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

@tf.function
def compute_loss_and_grads(feature_extractor, content_layer_name, content_weights, 
                        style_layer_names, combination_image, base_image, style_reference_image):
    
    with tf.GradientTape() as tape:
        loss = compute_loss(feature_extractor, content_layer_name, content_weight, 
        style_layer_names, combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

def get_vgg19():
    # Build a VGG19 model loaded with pre-trained ImageNet weights
    model = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # VGG19 (as a dict).
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

    style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_layer_name = "block5_conv2"

    return (feature_extractor, style_layer_names, content_layer_name)

def train_model(base_image_path, style_reference_image_path, iterations=400):

    # get pretrained network
    (feature_extractor, style_layer_names, content_layer_name) = get_vgg19()
    # optimizer
    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )

    # image transformation (base and style) to match the pretrained network
    base_image = preprocess_image(base_image_path)
    style_reference_image = preprocess_image(style_reference_image_path)
    # output: composed image 
    combination_image = tf.Variable(preprocess_image(base_image_path))

    for i in range(1, iterations + 1):
        # calculate grad and apply to tensors
        loss, grads = compute_loss_and_grads(
            feature_extractor, content_layer_name, content_weight, style_layer_names,
            combination_image, base_image, style_reference_image
        )
        # apply gradients to vars (i.e. combination_image)
        optimizer.apply_gradients([(grads, combination_image)])
        
        # report performance and save image
        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            fname = "iteration_%d.png" % i
            keras.preprocessing.image.save_img(fname, img)

def get_image_dimension(base_image_path):
    "get base image dimension "

    # Dimensions of the generated picture.
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    return img_nrows, img_ncols, width, height

if __name__ == '__main__':
    # This is the path to the image you want to transform.
    base_image_path = './data/nestor.jpeg'
    # This is the path to the style image.
    style_reference_image_path = './data/starry_night.jpeg'

    # base_image_path = keras.utils.get_file("paris.jpeg", "file:/home/fra/Project/pyProj/model_comparison/style_transfer/data")
    # style_reference_image_path = keras.utils.get_file("./data/starry_night.jpeg", None)
    (img_nrows, img_ncols, width, height) = get_image_dimension(base_image_path)

    train_model(base_image_path, style_reference_image_path)