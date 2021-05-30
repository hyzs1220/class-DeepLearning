# Experiment 6 MLP

## Experiment 

+ 因为在之前的实验中已经实现过线性分类器等一些比较简单的模型样例，所以在本次的实验中，一些代码实现还是比较简单的
+ 首先实现数据处理部分代码

```python
def dataset(path, ratio):
    N = 11

    # 读取数据
    data_pd = pd.read_csv(path, header=0, sep=';')
    data = []
    #  quality (score between 0 and 10)
    labels = []
    # 把输出分割出来
    for _, row in data_pd.iterrows():
        data.append([row['fixed acidity'], row['volatile acidity'], row['citric acid'],
                row['residual sugar'], row['chlorides'], row['free sulfur dioxide'],
                row['total sulfur dioxide'], row['density'], row['pH'],
                row['sulphates'], row['alcohol']])
        labels.append(row['quality'])
    print(data[1],labels[1])


    # 打乱顺序进行训练和测试
    # random.shuffle(data)
    # print(len(data),len(data[0]))
    # print(data)

    # 划分训练数据和测试数据
    count = len(data)
    split_point = int(count * ratio)
    train_data = data[:split_point]
    train_lables = labels[:split_point]
    test_data = data[split_point:]
    test_lables = labels[split_point:]
    print('There {} datas in total, {} datas used for train, {} used for test'.format(count, split_point, count - split_point))
    return train_data, train_lables, test_data, test_lables
```

+ 后面的四层MLP实现，参考了网上的一些[代码例子](https://gitee.com/hdt3213/ClassifiersDemo/blob/master/BPNN/bpnn.py)
+ 在具体实现中，就是按照多层感知机的模型结构进行构建，使用sigmoid激活函数，实现正向传播和反向梯度传播优化

```python
class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.hidden2_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.hidden2_cells = []
        self.output_cells = []
        self.input_weights = []
        self.hidden_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, nh2, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.hidden2_n = nh2
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.hidden2_cells = [1.0] * self.hidden2_n
        self.output_cells = [1] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.hidden_weights = make_matrix(self.hidden_n, self.hidden2_n)
        self.output_weights = make_matrix(self.hidden2_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for i in range(self.hidden_n):
            for h in range(self.hidden2_n):
                self.hidden_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.hidden_correction = make_matrix(self.hidden_n, self.hidden2_n)
        self.output_correction = make_matrix(self.hidden2_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate hidden2 layer
        for j in range(self.hidden2_n):
            total = 0.0
            for i in range(self.hidden_n):
                total += self.hidden_cells[i] * self.hidden_weights[i][j]
            self.hidden2_cells[j] = sigmoid(total)

        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden2_n):
                total += self.hidden2_cells[j] * self.output_weights[j][k]
            # self.output_cells[k] = sigmoid(total)
            self.output_cells[k] = int(total)

        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden2 layer error
        hidden2_deltas = [0.0] * self.hidden2_n
        for h in range(self.hidden2_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden2_deltas[h] = sigmoid_derivative(self.hidden2_cells[h]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.hidden2_n):
                error += hidden2_deltas[o] * self.hidden_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden2_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update hidden weights
        for h in range(self.hidden_n):
            for o in range(self.hidden2_n):
                change = hidden2_deltas[o] * self.hidden_cells[h]
                self.hidden_weights[h][o] += learn * change + correct * self.hidden_correction[h][o]
                self.hidden_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=200, correct=0.5):
        for iteration in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                if (type(label).__name__ != 'list'):
                    label = [label]
                case = cases[i]
                # print(case, label)
                error += self.back_propagate(case, label, learn, correct)
            if iteration % 10 == 0:
                print(f'Epoch {iteration} finished, globalError is {error}')
    
    def my_train(self, train_data, train_lables):
        self.setup(11, 20, 10, 1)
        self.train(train_data, train_lables, 1000, 0.001, 0.1)

    
    def my_test(self, test_data, test_lables):
        total = 0
        for i in tqdm(range(len(test_data))):
            label = test_lables[i]
            if (type(label).__name__ != 'list'):
                label = [label]
            case = test_data[i]
            # tmp = self.predict(case)
            # print(tmp, label, self.output_n,self.output_cells, error)
            for o in range(self.output_n):
                # error += label[o] - self.output_cells[o]
                if label[o] == self.output_cells[o]:
                    total += 1
        print('There {} test datas in total, accuracy is {} %'.format(len(test_data), (total / len(test_data))*100 ))


if __name__ == '__main__':
    train_data, train_lables, test_data, test_lables = dataset('winequality-red.csv', 0.8)
    nn = BPNeuralNetwork()
    # nn.test()
    nn.my_train(train_data, train_lables)
    nn.my_test(test_data, test_lables)
```

+ 最终看到四层MLP的效果和之前实验中实现的k-NN效果十分接近，不知道是不是因为一些实现细节上问题

+ 不过在实验中也能够看到

  + 随着训练轮数的增多，其效果在不断得到提升
  + 增加MLP隐藏层的节点数，也可以提高模型效果

+ ![](D:\study\仲国强-深度学习\exp6\exp6-MLP\1.png)

  

## Conclusion

+ 在实验中，因为一些别的安排，所以实验时间比较紧张，在代码实现并没有十分规范，可读性比较差，感觉是可以进一步优化封装的，所以在本次实验之后也会再去进行完善修改

  