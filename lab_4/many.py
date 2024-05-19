import multiprocessing as mp
from sklearn import preprocessing, svm, metrics
import numpy as np
import psutil
import json

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

    pred = linear_svm.predict(testing_norm)
    accuracy = metrics.accuracy_score(testing_labels, pred)
    f1_score = metrics.f1_score(testing_labels, pred)

    sorted_words = [bow.words[i] for i in np.argsort(linear_svm.coef_[0])]

    most_negative = sorted_words[:10]
    most_positive = sorted_words[-10:]

    return accuracy, f1_score, most_negative, most_positive

def nice():
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)

def main():
    training_data = np.load('data/training_sentences.npy', allow_pickle=True)
    training_labels = np.load('data/training_labels.npy')
    testing_data = np.load('data/test_sentences.npy', allow_pickle=True)
    testing_labels = np.load('data/test_labels.npy')

    bow = BagOfWords()
    bow.build_vocabulary(training_data)

    print(f"Number of words in training data: {len(bow.words)}")

    fixed_config_part = (training_data, training_labels, testing_data, testing_labels, bow)
    configs_dynamic_part = []
    for norm in ['l1', 'l2', 'standard', None]:
        for c in np.concatenate([np.linspace(0, 10, 50)[1:], np.linspace(20, 100, 50)]):
            configs_dynamic_part.append((norm, {"C": c, "kernel": "linear"}))

    configs = [(*fixed_config_part, *config) for config in configs_dynamic_part]

    with mp.Pool(max(mp.cpu_count() - 2, 1), nice) as p:
        results = p.starmap(run, configs)

    for i, (accuracy, f1_score, most_negative, most_positive) in enumerate(results):
        print(f"Configuration {i + 1}")
        print(f"Normalization: {configs[i][-2]}")
        print(f"SVM Config: {configs[i][-1]}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1_score}")
        print(f"Most negative words: {most_negative}")
        print(f"Most positive words: {most_positive}")
        print()

    results_with_configs = list(zip(results, configs_dynamic_part))

    with open('results.json', 'w', encoding='ascii') as f:
        json.dump(results_with_configs, f)

if __name__ == "__main__":
    main()
