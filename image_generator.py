import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator:

    def __init__(self, config):
        self.width = int(config["image_width"])
        self.noise_fraction = config["noise_fraction"]
        self.create_image = [self.create_rectangle, self.create_cross, self.create_disk, self.create_triangle] # map from image type to image function

    
    # For every pixel in an image, flip pixels with XOR by random chance
    def add_noise(self, image):
        for i in range(self.width):
            for j in range(self.width):
                if np.random.rand() < self.noise_fraction:
                    image[i, j] ^= 1
        return image

    
    # Create an image by generating two random corners of a rectangle
    # Then fill the edges
    def create_rectangle(self, min_width=4):
        image = np.zeros((self.width, self.width), dtype=int)
        upper_left_x = np.random.randint(0, self.width - min_width) # low inclusice, high exclusive
        upper_left_y = np.random.randint(0, self.width - min_width)
        lower_right_x = np.random.randint(upper_left_x + min_width - 1, self.width)
        lower_right_y = np.random.randint(upper_left_y + min_width - 1, self.width)
        for i in range(upper_left_x, lower_right_x + 1):
            image[upper_left_y, i] = 1
            image[lower_right_y, i] = 1
        for i in range(upper_left_y, lower_right_y + 1):
            image[i, upper_left_x] = 1
            image[i, lower_right_x] = 1
        return image


    # Generate a random center
    # Then generate a random length for each of the four spikes and fill it up
    def create_cross(self, min_width=2, max_width=10):
        image = np.zeros((self.width, self.width), dtype=int)
        center_x = np.random.randint(min_width, self.width - min_width)
        center_y = np.random.randint(min_width, self.width - min_width)
        image[center_y, center_x] = 1
        left = np.random.randint(min_width, max_width + 1)
        up = np.random.randint(min_width, max_width + 1)
        right = np.random.randint(min_width, max_width + 1)
        down = np.random.randint(min_width, max_width + 1)
        for i in range(self.width):
            if (i < center_x and i >= center_x - left - 1) or (i > center_x and i <= center_x + right):
                image[center_y, i] = 1
            if (i < center_y and i >= center_y - up - 1) or (i > center_y and i <= center_y + down):
                image[i, center_x] = 1
        return image


    # Generate a random center of the disk
    # Then loop over every point in the image, and check if the point is inside the radius
    def create_disk(self, min_radius=2):
        image = np.zeros((self.width, self.width), dtype=int)
        radius = np.random.randint(min_radius, self.width // 2)
        center_x = np.random.randint(radius + 1, self.width - radius)
        center_y = np.random.randint(radius + 1, self.width - radius)
        for i in range(self.width):
            for j in range(self.width):
                circle_value = (i - center_y) ** 2 + (j - center_x) ** 2
                if circle_value <= radius ** 2:
                    image[i, j] = 1
        return image

    
    # Generate a random corner and a width of both sides from the corner
    # Also create two variables that say wether or not to flip the triangle in either the vertical or horizontal direction
    # If the distance from the center of the triangle to the edge of the image is smaller than the width, also flip it
    # Then fill in both lengths of the triangle
    # For the diagonal, check the direction and fill in as a linear function
    def create_triangle(self, min_width=3, max_width=5):
        image = np.zeros((self.width, self.width), dtype=int)
        corner_x = np.random.randint(0, self.width)
        corner_y = np.random.randint(0, self.width)
        width = np.random.randint(min_width, max_width + 1)
        x_flip = False
        y_flip = False

        if corner_x + width >= self.width or (corner_x - width >= 0 and np.random.rand() > 0.5):
            x_flip = True
            for i in range(corner_x - width, corner_x + 1):
                image[corner_y, i] = 1
        else:
            for i in range(corner_x, corner_x + width + 1):
                image[corner_y, i] = 1
        if corner_y + width >= self.width or (corner_y - width >= 0 and np.random.rand() > 0.5):
            y_flip = True
            for i in range(corner_y - width, corner_y + 1):
                image[i, corner_x] = 1
        else:
            for i in range(corner_y, corner_y + width + 1):
                image[i, corner_x] = 1

        if x_flip and y_flip:
            for i in range(width):
                image[corner_y - i, corner_x - width + i] = 1
        elif not x_flip and y_flip:
            for i in range(width):
                image[corner_y - i, corner_x + width - i] = 1
        elif x_flip and not y_flip:
            for i in range(width):
                image[corner_y + i, corner_x - width + i] = 1
        elif not x_flip and not y_flip:
            for i in range(width):
                image[corner_y + i, corner_x + width - i] = 1
        return image

    
    # Check if images should be flatten or not and generate the shapes accordingly
    # Also check one-hot
    # Then for every image, generate a random image type and create that type
    # Add the noise afterwards
    def create_images(self, amount, flatten, one_hot=True):
        if flatten:
            images = np.zeros((amount, self.width * self.width))
        else:
            images = np.zeros((amount, self.width, self.width))

        if one_hot:
            labels = np.zeros((amount, 4))
        else:
            labels = np.zeros(amount)

        for i in range(amount):
            image_type = np.random.randint(0, 4)
            image = self.create_image[image_type]()
            image = self.add_noise(image)
            if flatten:
                image = image.flatten()
            images[i] = image

            if one_hot:
                labels[i, image_type] = 1
            else:
                labels[i] = image_type
        
        return images, labels
    

    # The data is already randomly generated, so no need to further shuffle
    # Split the data into three sets with the given ratios
    def train_val_test_split(self, data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        train = data[0 : int(data.shape[0] * train_ratio)]
        val = data[int(data.shape[0] * train_ratio) : int(data.shape[0] * (train_ratio + val_ratio))]
        test = data[int(data.shape[0] * (train_ratio + val_ratio)) : data.shape[0]]
        return train, val, test


    # Generate images and labels, and then split them into train, val and test sets
    def generate(self, amount, flatten):
        images, labels = self.create_images(amount, flatten)
        x_train, x_val, x_test = self.train_val_test_split(images)
        y_train, y_val, y_test = self.train_val_test_split(labels)
        return (x_train, x_val, x_test), (y_train, y_val, y_test)

    
    # Generate images, and plot them with matplotlib
    def display(self, amount):
        images, labels = self.create_images(amount, False, False)
        image_types = {0: "Rectangle", 1: "Cross", 2: "Disk", 3: "Triangle"}
        for i in range(amount):
            plt.imshow(images[i])
            plt.title(image_types[labels[i]])
            ax = plt.gca()
            ax.set_xticks(np.arange(0, self.width))
            ax.set_yticks(np.arange(0, self.width))
            ax.set_xticks(np.arange(-0.5, self.width), minor=True)
            ax.set_yticks(np.arange(-0.5, self.width), minor=True)
            ax.grid(which="minor")
            plt.show()

