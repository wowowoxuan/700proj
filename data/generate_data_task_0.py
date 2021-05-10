import random
import numpy as np

def generate_training_data(num_samples):
    training_set = []
    for i in range(num_samples):
        print(i)
        length = random.randint(1,10)
        print(length)
        temp_list = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        print(temp_list)
        training_set.append(temp_list)
    np.save('./train.npy',np.array(training_set))
    return np.array(training_set)





def generate_testing_data(max_len, num_samples):
    training_set = []
    for i in range(num_samples):
        print(i)
        length = random.randint(1,max_len)
        temp_list = []
        for j in range(length):
            idx = random.randint(0,25)
            onehot = [0] * 26
            onehot[idx] = 1
            temp_list.append(onehot)
        training_set.append(temp_list)
    np.save('./val.npy',np.array(training_set))
    return np.array(training_set)
train = generate_training_data(10000)
val = generate_testing_data(50,5000)
print(train.shape)
print(val.shape)

# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# a = np.load('./val.npy')
# # print(a==b)
# a = a.tolist()
# print(len(a))