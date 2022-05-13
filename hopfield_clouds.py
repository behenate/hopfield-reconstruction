import io
from os import listdir
from os.path import isfile, join

from PIL import Image
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import images2xi
import numpy as np
import matplotlib.pyplot as plt


class HopfieldClouds:
    def __init__(self, size):
        #N - number of neurons
        self.N = size
        self.hopfield_network = HopfieldNetwork(N=self.N)
        self.current_image_index = 0

        train_files = [f for f in listdir('chmurki') if isfile(join('chmurki', f))]
        self.train_paths = [join('chmurki', f) for f in train_files]

        print("Loading images...")
        self.xi = images2xi(self.train_paths, self.N)
        print("Training network...")
        self.hopfield_network.train_pattern(self.xi)

    # Get an image from self.xi with provided index, remove provided percentage of pixels and try to reconstruct
    # Returns the cropped image and the reconstructed image
    def predict_image(self, index, obstruction_percentage=20, iterations=1):
        half_image = np.copy(self.xi[:, index])
        half_image[: int(self.N / (100 / obstruction_percentage))] = -1
        self.hopfield_network.set_initial_neurons_state(np.copy(half_image))
        self.hopfield_network.update_neurons(iterations, 'async')

        image_pil = self.__xi_to_PIL(self.hopfield_network.S)
        half_image_pil = self.__xi_to_PIL(half_image)
        return half_image_pil, image_pil

    # Utility function, which changes xi from the hopfield network library to a PIL image
    def __xi_to_PIL(self, xi):
        N_sqrt = int(np.sqrt(self.hopfield_network.N))
        image_arr = np.uint8((xi.reshape(N_sqrt, N_sqrt) + 1) * 90)
        plt.matshow(image_arr, cmap="Blues")
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        im = Image.open(img_buf)
        # img_buf.close()
        return im

    def get_current_cropped(self, obstruction_percentage=20):
        half_image = np.copy(self.xi[:, self.current_image_index])
        half_image[: int(self.N / (100 / obstruction_percentage))] = -1
        half_image_pil = self.__xi_to_PIL(half_image)
        return half_image_pil


    # Moves the current index to the next image and returns it
    def next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.train_paths)
        return self.get_current_image()

    # Moves the current index to the previous image and returns it
    def prev_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.train_paths)
        return self.get_current_image()

    # Returns the current image
    def get_current_image(self):
        image = np.copy(self.xi[:, self.current_image_index])
        return self.__xi_to_PIL(image)

    # Returns the predictions for the current image with the provided obstruction percentage
    def get_current_image_predictions(self, obstruction_percentage=20, iterations=1):
        return self.predict_image(self.current_image_index, obstruction_percentage, iterations)
