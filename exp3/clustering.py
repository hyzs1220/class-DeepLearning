import pandas as pd
from collections import Counter


data_pd = pd.read_csv('winequality-red.csv', header=0, sep=';')

X = []
C = []


for _, row in data_pd.iterrows():
    X.append([row['fixed acidity'], row['volatile acidity'], row['citric acid'],
              row['residual sugar'], row['chlorides'], row['free sulfur dioxide'],
              row['total sulfur dioxide'], row['density'], row['pH'],
              row['sulphates'], row['alcohol']])
    C.append(row['quality'])

data_size = len(X)

train_num = int(data_size * 0.8)
test_num = data_size - train_num

train_X = X[:train_num]
test_X = X[train_num:]

train_C = C[:train_num]
test_C = C[train_num:]

C_new = [0 for _ in range(test_num)]

k = 15

correct_num = 0

for i, x_i in enumerate(test_X):
    distance_c_list = []
    for j, x_j in enumerate(train_X):
        distance = sum([(x_i[k] - x_j[k]) ** 2 for k in range(len(x_i))])
        distance_c_list.append((distance, train_C[j]))
    distance_c_list.sort(key=lambda x: x[0])

    nearest_k_labels = [distance_c[1] for distance_c in distance_c_list[:k]]
    count_result = Counter(nearest_k_labels)

    new_c = list(count_result.keys())[0]
    if new_c == test_C[i]:
        correct_num += 1
    C_new[i] = new_c


print(correct_num / test_num)


# result = [train_C[i] - C_new[i] for i in range(int(data_size * 0.8))]

# print(result.count(0))

