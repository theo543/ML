import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def linearize(img):
    img = np.copy(img)
    img[img > 128] = 255
    img[img <= 128] = 0
    return img

def main():
    images = np.array([np.load(f"car_{i}.npy") for i in range(9)])
    img_1 = linearize(images[0])
    img_2 = linearize(images[1])
    io.imshow(img_1)
    io.show()
    io.imshow(img_2)
    io.show()

if __name__ == "__main__":
    main()
