import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

def values_to_bins(x, bins):
    return np.digitize(x, bins) - 1

#def values_to_bins(x, bins):
#    data = np.digitize(x, bins) - 1
#    res = np.zeros((x.shape[0], 14, 14))
#    data = data.reshape((x.shape[0], 28, 28))
#    for s in range(x.shape[0]):
#        for i in range(28):
#            for j in range(28):
#                res[s, i // 2, j // 2] += data[s, i, j]
#    return res.reshape(x.shape[0], 14 * 14)


def main():
    train_images_original = np.loadtxt('train_images.txt')
    train_labels = np.loadtxt('train_labels.txt').astype(np.int64)
    test_images_original = np.loadtxt('test_images.txt')
    test_labels = np.loadtxt('test_labels.txt').astype(np.int64)
    best_bins = 0
    best_score = -1
    for num_bins in list(range(1, 10)) + list(range(10, 255 + 1, 10)) + [255]:
        bins = np.linspace(start=0, stop=255, num=num_bins)
        train_images = values_to_bins(train_images_original, bins)
        test_images = values_to_bins(test_images_original, bins)
        naive_bayes = MultinomialNB()
        naive_bayes.fit(train_images, train_labels)
        accuracy = naive_bayes.score(test_images, test_labels)
        print(f"Accuracy with {num_bins} bins: {accuracy}")
        if accuracy > best_score:
            best_score = accuracy
            best_bins = num_bins
    print(f"Best number of bins: {best_bins}")
    bins = np.linspace(start=0, stop=255, num=best_bins)
    train_images = values_to_bins(train_images_original, bins)
    test_images = values_to_bins(test_images_original, bins)
    naive_bayes = MultinomialNB()
    naive_bayes.fit(train_images, train_labels)
    accuracy = naive_bayes.score(test_images, test_labels)
    print(f"Final accuracy: {accuracy}")
    pred = naive_bayes.predict(test_images)
    confusion_matrix = np.zeros((10, 10))
    for label, pr in zip(test_labels, pred):
        confusion_matrix[label, pr] += 1
    print(confusion_matrix)
    is_misclassified = pred != test_labels
    misclassified_img = test_images_original[is_misclassified]
    misclassified_labels = test_labels[is_misclassified]
    wrong_pred = pred[is_misclassified]
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(misclassified_img[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {wrong_pred[i]}\nCorrect: {misclassified_labels[i]}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
