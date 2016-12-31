import numpy as np

def importdata(path_to_train ,path_to_test):
    train_file = open(path_to_train,'r')
    test_file = open(path_to_test,'r')

    tr_data = np.array([])
    test_data = np.array([])

    for line in train_file:
        data_point_str = line.split()
        data_point_float = []
        for d in data_point_str:
            data_point_float.append(float(d))

        # check number of data points in tr_data
        if tr_data.size == 0:
            tr_data = np.array(data_point_float[0:2])
        else:
            tr_data = np.vstack((tr_data,data_point_float[0:2]))

    train_file.close()

    for line in test_file:
        data_point_str = line.split()
        data_point_float = []
        for d in data_point_str:
            data_point_float.append(float(d))

        # check number of data points in tr_data
        if test_data.size == 0:
            test_data = np.array(data_point_float[0:2])
        else:
            test_data = np.vstack((test_data,data_point_float[0:2]))

    test_file.close()

    return tr_data, test_data