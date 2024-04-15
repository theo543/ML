import numpy as np
from skimage import io

def main():
    images = np.array([np.load(f"car_{i}.npy") for i in range(9)])
    sum_all_pixels = np.sum(images)
    print(f"sum_all_pixels: {sum_all_pixels}")
    sum_all_pixels_each_image = np.sum(images, axis=(1, 2))
    print(f"sum_all_pixels_each_image: {sum_all_pixels_each_image}")
    greatest_sum_index = np.argmax(sum_all_pixels_each_image)
    print(f"greatest_sum_index: {greatest_sum_index}")
    avg_image = np.sum(images, axis=0) / images.shape[0]
    #io.imshow(avg_image.astype(np.uint8))
    #io.show()
    standard_dev = np.std(images)
    #images_normalized = (images - avg_image) / standard_dev
    for i in range(9):
        gradient_x, gradient_y = np.gradient(images[i, :, :])
        gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        io.imshow(gradient.astype(np.uint8))
        io.show()
        io.imshow(images[i].astype(np.uint8))
        io.show()
        patch_width = 5
        patch_height = 5
        best_gradient_sum = np.NINF
        (best_x, best_y) = (0, 0)
        for x in range(0, images.shape[1] - patch_width):
            for y in range(0, images.shape[2] - patch_height):
                total_gradient = np.sum(gradient[x:x + patch_width, y:y + patch_height])
                if total_gradient > best_gradient_sum:
                    best_gradient_sum = total_gradient
                    (best_x, best_y) = (x, y)
        print(f"Gradient of result = {best_gradient_sum}")
        io.imshow(images[i, best_x:best_x + patch_width, best_y:best_y + patch_height].astype(np.uint8))
        io.show()

if __name__ == "__main__":
    main()
