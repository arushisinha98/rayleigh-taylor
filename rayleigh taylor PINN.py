import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def save_fig(outfile, files, fps = 5, loop = 1):
  """ Function to save GIFs """
  imgs = [Image.open(file) for file in files]
  imgs[0].save(fp = outfile, format = 'GIF', append_images = imgs[1:], save_all = True,
               duration = int(1000/fps), loop = loop)

