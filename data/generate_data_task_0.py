import random
import numpy as np

def generate_training_data(num_samples):
    training_set = []
    for i in range(num_samples):
        length = random.randint(1,10)
        temp_list = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        training_set.append(temp_list)
    np.save('./test.npy',np.array(training_set))
    return np.array(training_set)





def generate_testing_data(length, num_samples):
    return 'not implemented'
b=generate_training_data(2)
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
a = np.load('./test.npy')
print(a==b)
a = a.tolist()
print(a[1])