import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def tensor_to_image(tensor):
  '''converts a tensor to an image'''
  tensor_shape = tf.shape(tensor)
  number_elem_shape = tf.shape(tensor_shape)
  if number_elem_shape > 3:
    assert tensor_shape[0] == 1
    tensor = tensor[0]
  return tf.keras.preprocessing.image.array_to_img(tensor)


def load_img(path_to_img):
  '''loads an image as a tensor and scales it to 512 pixels'''
  max_dim = 512
  image = tf.io.read_file(path_to_img)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)

  shape = tf.shape(image)[:-1]
  shape = tf.cast(tf.shape(image)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  image = tf.image.resize(image, new_shape)
  image = image[tf.newaxis, :]
  image = tf.image.convert_image_dtype(image, tf.uint8)

  return image


def load_images(content_path, style_path):
  '''loads the content and path images as tensors'''
  content_image = load_img("{}".format(content_path))
  style_image = load_img("{}".format(style_path))

  return content_image, style_image


def imshow(image, title=None):
  '''displays an image with a corresponding title'''
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


def show_images_with_objects(images, titles=[]):
  '''displays a row of images with corresponding titles'''
  if len(images) != len(titles):
    return

  plt.figure(figsize=(20, 12))
  for idx, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(1, len(images), idx + 1)
    plt.xticks([])
    plt.yticks([])
    imshow(image, title)



def save_img(img_tensor, file_path):
    '''Saves an image tensor to a file'''
    img_tensor = tf.keras.preprocessing.image.img_to_array(img_tensor)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.uint8)
    img_pil = Image.fromarray(np.array(img_tensor))
    img_pil.save(file_path)


content_path = './images/congchua.png'
style_path = './images/Vassily_Kandinsky.jpg'

content_image, style_image = load_images(content_path, style_path)
show_images_with_objects([content_image, style_image],
                         titles=[f'content image: {content_path}',
                                 f'style image: {style_path}'])

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2') 

stylized_image = hub_module(tf.image.convert_image_dtype(content_image, tf.float32),
                            tf.image.convert_image_dtype(style_image, tf.float32))[0]

# convert the tensor to image
tensor_to_image(stylized_image)

def save_img(img_tensor, file_path):
    '''Saves an image tensor to a file'''
    # Squeeze the batch dimension if present
    img_tensor = tf.squeeze(img_tensor, axis=0)

    # Convert image tensor to NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(img_tensor)

    # Convert to uint8 and create PIL Image
    img_array = tf.image.convert_image_dtype(img_array, tf.uint8)
    img_pil = Image.fromarray(np.array(img_array))

    # Save the image to the specified file path
    img_pil.save(file_path)
    
save_img(stylized_image, "output.jpg")


