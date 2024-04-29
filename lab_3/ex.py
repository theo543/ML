import numpy as np
from pathlib import Path
from einops import rearrange

def cos_sim(a, b):
    div = np.linalg.norm(a) * np.linalg.norm(b)
    if div == 0:
        return 0
    return np.dot(a, b) / div

def preprocess_image(image):
    image = image.reshape((28, 28))
    image = rearrange(image, "(p1 h) (p2 w) -> p1 p2 h w", p1 = 7, p2 = 7)
    similarity = []
    for j in range(7):
        for i in range(j + 1, 7):
            similarity.append(cos_sim(image[i][j].flatten(), image[7 - i - 1][7 - j - 1].flatten()))
    return np.array(similarity)

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'):
        if metric == 'l1':
            dist = np.sum(np.abs(self.train_images - test_image), axis=1)
        else:
            dist = np.sqrt(np.sum(np.square(self.train_images - test_image), axis=1))
        idx = np.argpartition(dist, num_neighbors)[:num_neighbors]
        labels = self.train_labels[idx]
        label, count = np.unique(labels, return_counts=True)
        ind = np.argmax(count)
        return label[ind]

def main():
    train_images = [preprocess_image(i) for i in np.loadtxt('train_images.txt')]
    train_labels = np.loadtxt('train_labels.txt').astype(np.int64)
    test_images = [preprocess_image(i) for i in np.loadtxt('test_images.txt')]
    test_labels = np.loadtxt('test_labels.txt').astype(np.int64)

    knn = KnnClassifier(train_images, train_labels)

    test_predictions = [knn.classify_image(image, num_neighbors=3) for image in test_images]
    np.savetxt("pred_3nn_l2_mnist.txt", test_predictions)
    print(test_predictions)
    correct = np.sum(np.equal(test_predictions, test_labels))
    print(f"{correct} / {len(test_labels)} = {correct / len(test_labels)}")

    for metric in ["l1", "l2"]:
        if Path(f"accuracy_{metric}.txt").exists():
            continue
        k_vals = [1, 3, 5, 7, 9]
        accuracy = []
        for k in k_vals:
            pred = [knn.classify_image(image, num_neighbors=k, metric=metric) for image in test_images]
            correct = np.sum(np.equal(pred, test_labels))
            accuracy.append(correct / len(test_images))
        np.savetxt(f"accuracy_{metric}.txt", accuracy)

if __name__ == "__main__":
    main()
