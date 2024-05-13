from sklearn import preprocessing, svm, metrics
import numpy as np

def normalize_data(train_data, test_data, norm=None):
    match norm:
        case 'standard':
            scaler = preprocessing.StandardScaler()
            scaler.fit(train_data)
            return scaler.transform(train_data), scaler.transform(test_data)
        case 'l1':
            def l1(data):
                scale = np.sum(np.abs(data), axis=1).reshape(-1, 1)
                return np.divide(data, scale, where=scale != 0)
            return l1(train_data), l1(test_data)
        case 'l2':
            def l2(data):
                scale = np.sqrt(np.sum(data ** 2, axis=1)).reshape(-1, 1)
                return np.divide(data, scale, where=scale != 0)
            return l2(train_data), l2(test_data)
        case None:
            return train_data, test_data
        case _:
            raise ValueError(f"Unknown normalization: {norm}")

class BagOfWords:
    def __init__(self):
        self.word_id = {}
        self.words = []

    def build_vocabulary(self, data):
        for sentence in data:
            for word in sentence:
                if word not in self.word_id:
                    self.word_id[word] = len(self.words)
                    self.words.append(word)

    def get_features(self, data):
        features = np.zeros((len(data), len(self.words)))
        for i, sentence in enumerate(data):
            for word in sentence:
                if word in self.word_id:
                    features[i, self.word_id[word]] += 1
        return features

def run(training_data, training_labels, testing_data, testing_labels, bow, norm, svm_config):
    training_norm, testing_norm = normalize_data(bow.get_features(training_data), bow.get_features(testing_data), norm)

    linear_svm = svm.SVC(**svm_config)
    linear_svm.fit(training_norm, training_labels)

    accuracy = linear_svm.score(testing_norm, testing_labels)
    f1_score = metrics.f1_score(testing_labels, linear_svm.predict(testing_norm))

    print(f"Accuracy: {accuracy}\nF1 Score: {f1_score}")

    sorted_words = [bow.words[i] for i in np.argsort(linear_svm.coef_[0])]

    print(f"10 most negative words: {sorted_words[:10]}")
    print(f"10 most positive words: {sorted_words[-10:]}")

def main():
    norm = 'l2'
    svm_config = {"C": 1.0, "kernel": "linear"}

    training_data = np.load('data/training_sentences.npy', allow_pickle=True)
    training_labels = np.load('data/training_labels.npy')
    testing_data = np.load('data/test_sentences.npy', allow_pickle=True)
    testing_labels = np.load('data/test_labels.npy')

    bow = BagOfWords()
    bow.build_vocabulary(training_data)

    print(f"Number of words in training data: {len(bow.words)}")

    run(training_data, training_labels, testing_data, testing_labels, bow, norm, svm_config)

if __name__ == "__main__":
    main()
