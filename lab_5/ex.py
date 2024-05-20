import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def normalize(train_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler.transform(train_data), scaler.transform(test_data)

def train(train_data, train_labels, test_data, test_labels, model_class):
    linear_regression = model_class()
    linear_regression.fit(train_data, train_labels)
    predicted = linear_regression.predict(test_data)
    mse = mean_squared_error(test_labels, predicted)
    mae = mean_absolute_error(test_labels, predicted)
    return mse, mae

def train_linear_regression(train_data, train_labels, test_data, test_labels):
    return train(train_data, train_labels, test_data, test_labels, LinearRegression)

def train_ridge(train_data, train_labels, test_data, test_labels, alpha):
    return train(train_data, train_labels, test_data, test_labels, lambda: Ridge(alpha=alpha))

def main():
    training_data = np.load('data/training_data.npy')
    prices = np.load('data/prices.npy')
    resample(training_data, prices, replace=False, random_state=0)

    data_len = prices.shape[0]
    fold_len = data_len // 3
    fold_1 = slice(0, fold_len)
    fold_2 = slice(fold_len, 2 * fold_len)
    fold_3 = slice(2 * fold_len, data_len)
    folds = [(fold_1, (fold_2, fold_3)), (fold_2, (fold_1, fold_3)), (fold_3, (fold_1, fold_2))]

    def assemble_data(held_back, train_1, train_2):
        train_data = np.concatenate((training_data[train_1], training_data[train_2]))
        train_prices = np.concatenate((prices[train_1], prices[train_2]))
        test_data = training_data[held_back]
        test_prices = prices[held_back]
        train_data, test_data = normalize(train_data, test_data)
        return train_data, train_prices, test_data, test_prices

    mse = []
    mae = []
    for (held_back, (train_1, train_2)) in folds:
        train_d, train_l, test_d, test_l = assemble_data(held_back, train_1, train_2)
        mse_, mae_ = train_linear_regression(train_d, train_l, test_d, test_l)
        mse.append(mse_)
        mae.append(mae_)

    print(f"Mean Squared Error average for linear regression: {np.mean(mse)}")
    print(f"Mean Absolute Error average for linear regression: {np.mean(mae)}")

    best_alpha = -1
    best_alpha_mse = float('inf')
    for alpha in [1, 10, 100, 1000]:
        mse = []
        mae = []
        for (held_back, (train_1, train_2)) in folds:
            train_d, train_l, test_d, test_l = assemble_data(held_back, train_1, train_2)
            mse_, mae_ = train_ridge(train_d, train_l, test_d, test_l, alpha)
            mse.append(mse_)
            mae.append(mae_)
        print(f"Mean Squared Error average for ridge with alpha={alpha}: {np.mean(mse)}")
        mse = np.mean(mse)
        if mse < best_alpha_mse:
            best_alpha = alpha
            best_alpha_mse = mse

    print(f"Best alpha for ridge: {best_alpha} with MSE: {best_alpha_mse}")

    big_ridge = Ridge(alpha=best_alpha)
    big_ridge.fit(training_data, prices)
    print(big_ridge.coef_)
    print(big_ridge.intercept_)
    # show most significant features
    feature_names = ["Year", "Kilometers", "Mileage", "Engine", "Power", "Seats", "Owners",
                     *[f"Has Fuel Type {i}" for i in range(1, 5 + 1)],
                     "Is Manual", "Is Automatic"]
    features = np.argsort(np.abs(big_ridge.coef_))
    most_significant = [feature_names[i] for i in features]
    print(f"Most significant: {most_significant[0]}")
    print(f"Second most significant: {most_significant[1]}")
    print(f"Least significant: {most_significant[-1]}")
    print(f"Features sorted by significance: {most_significant}")

if __name__ == "__main__":
    main()
