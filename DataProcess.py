import os
import wfdb
import numpy as np

label_list = [1, 2, 3, 8, 4, 7, 9, 5, 6, 31, 34, 11, 10, 12, 38, 24, 13]


def process_data(window_seconds=10, fs=360):
    print("Processing data...")
    raw_dir = "./mitdb/"
    # processed_dir = "./processed/"

    ecg_dir = os.listdir(raw_dir)
    # if not os.path.exists(processed_dir):
    #     os.makedirs(processed_dir)

    X = []
    Y = []
    for ecg_name in ecg_dir:
        if ecg_name.endswith(".dat"):
            print("Processing {}".format(ecg_name))
            name = ecg_name.replace(".dat", '')

            # Read directory
            raw_data = raw_dir + '/' + name

            p_signal = wfdb.rdrecord(raw_data).p_signal
            ann = wfdb.rdann(raw_data, 'atr', return_label_elements='label_store')
            label = ann.label_store
            label_pos = ann.sample

            window_size = window_seconds * fs

            print('len:', len(label))
            i = 0
            while i < len(label):
                while i < len(label) and label[i] not in label_list:
                    i += 1
                cur_label = label[i]
                front_label_pos = label_pos[i]
                rear_label_pos = label_pos[i]
                i += 1
                while rear_label_pos - front_label_pos < window_size and front_label_pos + window_size < p_signal.shape[
                        0] and i < len(label):
                    next_label = label[i]
                    if cur_label != next_label:
                        cur_label = next_label
                        front_label_pos = label_pos[i]
                    rear_label_pos = label_pos[i]
                    i += 1

                if front_label_pos + window_size < p_signal.shape[0]:
                    x = p_signal[front_label_pos:front_label_pos + window_size, :]
                    y = label_list.index(cur_label)
                    X.append(x)
                    Y.append(y)
                else:
                    break
    X = np.array(X)
    Y = np.array(Y)
    np.savez_compressed('./data.npz', X=X, Y=Y)


if __name__ == "__main__":
    process_data()
