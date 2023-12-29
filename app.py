# Import necessary libraries
import gradio as gr
from flask import Flask, render_template, request
import os
from PIL import Image
import io
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Import the image-to-text pipeline from the transformers library
from transformers import pipeline
from utilities import *

# Load the style transfer model from TensorFlow Hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Define the neural style transfer function
def neural_style_transfer(content_image, style_image):
    # Convert the content and style images to float32 tensors
    content_image = tf.image.convert_image_dtype(content_image, tf.float32)[tf.newaxis, ...]
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)[tf.newaxis, ...]

    # Apply neural style transfer
    stylized_image = hub_module(content_image, style_image)[0]

    # Remove the batch dimension
    stylized_image = tf.squeeze(stylized_image, axis=0)

    # Convert the tensor to PIL Image
    stylized_image = tf.keras.preprocessing.image.array_to_img(stylized_image.numpy())
    return stylized_image


# Create a Flask web application
app = Flask(__name__)

# Define a Gradio Interface for image input and text output
iface = gr.Interface(
    fn=neural_style_transfer,
    inputs=[
        gr.Image(label="Upload content image", type="pil"),
        gr.Image(label="Upload style image", type="pil")
    ],
    outputs=[gr.Image(label="Styled Image")],
    title="Neural Style Transfer",
    description="Apply the style of one image to another using neural style transfer",
    allow_flagging="never",  # Disable user flagging for simplicity
    examples=[
        ["./images/lego-ninjago.jpg", "./images/Vassily_Kandinsky.jpg"],
        ["./images/congchua.png", "./images/Vassily_Kandinsky.jpg"]
    ]
)

# Define a route for the home page with handling of POST requests
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded content and style image files from the form
        content_image_file = request.files['content_image']
        style_image_file = request.files['style_image']

        if content_image_file and style_image_file:
            # Open the uploaded images as PIL images
            content_image = Image.open(content_image_file)
            style_image = Image.open(style_image_file)

            # Perform neural style transfer
            stylized_image = neural_style_transfer(content_image, style_image)

            # Save the stylized image or do further processing as needed
            stylized_image.save("static/stylized_image.jpg")

            # Render the result using the HTML template
            return render_template('index.html', stylized_image="stylized_image.jpg")

    # Render the home page template
    return render_template('index.html')

# Run the Flask web application
if __name__ == '__main__':
    # Launch the Gradio Interface in a separate thread with sharing enabled
    iface.launch(share=True)

    # Run the Flask web application with debugging enabled
    app.run(debug=True)

