import numpy as np
import matplotlib.pyplot as plt

def main():
    k_vals = [1, 3, 5, 7, 9]
    for metric in ['l1', 'l2']:
        accuracy = np.loadtxt(f"accuracy_{metric}.txt")
        plt.plot(k_vals, accuracy)
        plt.title(metric)
        plt.show()

if __name__ == "__main__":
    main()
