import numpy as np
from config_parser import parse_config
from network import Network
from image_generator import ImageGenerator

if __name__ == "__main__":

    #np.random.seed(0)

    config = parse_config("config3.txt")

    data_generator = ImageGenerator(config)

    if config["display"] == "true":
        display_amount = int(config["display_amount"])
        data_generator.display(display_amount)
        exit()
    
    verbose = config["verbose"] == "true"
    visualize_kernels = config["visualize_kernels"] == "true"

    image_amount = int(config["image_amount"])
    x, y = data_generator.generate(image_amount, True)
    x_train, x_val, x_test = x
    y_train, y_val, y_test = y

    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])

    network = Network(config)
    network.fit(x_train.T, y_train.T, x_val.T, y_val.T, x_test.T, y_test.T, epochs, batch_size, verbose, visualize_kernels)
