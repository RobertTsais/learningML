import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    """
    根据课上所学知识(致敬张军英老师)，实现了一个感知机，包括AND, OR, NOT, 可以使用正态分布的样本数据
    感知机原理可参考：https://zhuanlan.zhihu.com/p/29836398  https://zhuanlan.zhihu.com/p/42438035
    https://blog.csdn.net/u012759262/article/details/101943109
    """
    def __init__(self, kind='AND'):
        self.weights = np.array([0.2, 0.4], ndmin=2)  # 初始化权重，为图方便硬编码了
        self.bias = 0.1  # 初始化偏置，为图方便硬编码了
        self.inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # (m,2)矩阵 m:=dataSize，每个样本数据为二维，即2个特征,(f1, f2)
        if kind.upper() == 'AND':
            self.labels = np.array([[-1], [-1], [-1], [1]])  # AND  (m,1)矩阵
        elif kind.upper() == 'OR':
            self.labels = np.array([[-1], [1], [1], [1]]) # OR
        elif kind.upper() == 'NOT':
            # NOT, w2 cannot be zero, let w1 be zero, so have to make labels like this
            self.labels = np.array([[-1], [1], [-1], [1]])
        else:  # default is AND perceptron
            self.labels = np.array([[-1], [-1], [-1], [1]])  # AND  (m,1)矩阵
        self.dataSize = self.labels.size
        self.misclassifiedSet = np.array(np.zeros((self.dataSize, 1), dtype=int))  # 错分集标识初始化，(m,1)矩阵
        self.learningRate = 0.1  # adjust this when necessary
        self.errors = np.array([])  # errors in each iteration
        self.fig, self.axes = plt.subplots(1, 2)  # for plotting diagram
        self.initDiagram()
        self.firstPlot = True  # just for special usage

    def __str__(self):
        """
        输出训练后的感知机学习到的权重、偏置
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def initDiagram(self):
        self.fig.set_size_inches(10, 5)  # set figure window size (10 inches * 5 inches)
        # plt.figure(figsize=(10, 5))
        self.axes[0].set_title('Perceptron Demo')
        self.axes[0].axis('scaled')
        # self.axes.axis('square')
        self.axes[0].axis([-1, 2, -1, 2])
        self.axes[0].set_xlabel('feature 1')
        self.axes[0].set_ylabel('feature 2')
        self.axes[0].grid()

        # self.axes[1].axis('scaled')
        self.axes[1].set_title('Cost Curve')
        # self.axes[1].axis([0, 30, -10, 0.5])
        self.axes[1].set_xlabel('iteration times')
        self.axes[1].set_ylabel('errors', color="C0")
        self.axes[1].set_xticks(range(0, 50))
        self.axes[1].grid()

    def plotBoundary(self):
        """
        图形可视化，绘制样本点分布，绘制决策边界线，要和show()配合使用
        """
        # .T就是转置, *是numpy特有的，类似C++的指针解引用，这里是二维数组解引用，即降维。而且是先运算的.T然后解引用迭代器
        # plt.plot(*self.inputs.T, 'o')  # [0 0 1 1] [0 1 0 1]
        if self.firstPlot == True:
            for i in range(self.dataSize):
                if self.labels[i] == -1:
                    self.axes[0].plot(*self.inputs[i], 'c.')  # negative sample with marker style 'c.'
                elif self.labels[i] == 1:
                    self.axes[0].plot(*self.inputs[i], 'r+')  # positive sample with maker style 'r+'
        x1 = np.linspace(-1, 2, num=3)  # not necessary to calculate too many points for it's a straight line
        if self.weights[0, 1] != 0:
            x2 = (-self.bias * np.ones(3) - self.weights[0, 0] * x1) / self.weights[0, 1]
            self.axes[0].plot(x1, x2)  # plot decision boundary line
        costX = np.arange(0, self.errors.size, 1)
        print('costX:%s' % costX)
        print('costY:%s' % self.errors)
        self.axes[1].plot(costX, self.errors, 'k-')  # plot cost curve
        plt.pause(0.05)
        self.firstPlot = False  # try comment this line

    def show(self):
        plt.show()

    def adjLearningRate(self, learningRate):
        self.learningRate = learningRate

    def activation(self):
        z = self.weights.dot(self.inputs.T) + self.bias  # 激活输出Z = W·X^T+b  z.shape():=(1,m)
        return z.T  # (m,1)矩阵
        # return 1 if z >= 0 else -1

    def genMisclassifiedSet(self):
        # print('activation output:->\n%s\n<--' % self.activation())
        y = self.activation() * self.labels  # y still is (m,1)
        # print('activation multiplied by label:->\n%s\n<--' % y)
        self.misclassifiedSet.fill(0)  # remember to reset
        errSum = 0
        for i in range(self.dataSize):
            if y[i, 0] <= 0.1:  # criterion, adjust this when necessary
                errSum += y[i, 0]
                self.misclassifiedSet[i] = 1
        self.errors = np.append(self.errors, errSum)
        # print('misclassifiedSet:->\n%s\n<--' % self.misclassifiedSet)

    def training(self, iterationTimes=10):
        if iterationTimes == 10:
            iterationTimes = 50
        iterationCnt = 0
        self.genMisclassifiedSet()
        # until all samples are classified correctly, otherwise just iterate 50 times for it is linearly inseparable
        while self.misclassifiedSet.any() and iterationCnt < iterationTimes:
            iterationCnt += 1
            print('--iteration %d times--' % iterationCnt )
            for i in range(self.dataSize):
                if self.misclassifiedSet[i] == 1:
                    # print('--to update param for sample %s--' % self.inputs[i])
                    # print("it's label: " + str(self.labels[i]))
                    # fixed-increment and update once by every misclassified single-sample
                    # that is, a SGD(Stochastic Gradient Descent) algorithm
                    # Cost function: C(w, b) = -Σ y·(w·x + b)  sum on every misclassified sample
                    self.weights += self.learningRate * self.labels[i] * self.inputs[i]  # 核心部分，更新权重
                    # print('updated weights: ' + str(self.weights))
                    self.bias += self.learningRate * self.labels[i]  # 核心部分，更新偏置
                    # print('updated bias: ' + str(self.bias))
                    # print('--updated param for sample %s--' % self.inputs[i])
            self.genMisclassifiedSet()
            self.plotBoundary()  # 看过程
        # self.plotBoundary()  # 看结果
        self.show()

    def getTraningDataset(self, inputVec, labels):
        """
        支持传入外部训练集学习参数
        :param inputVec: (m,2)矩阵，即m个样本，每个样本2个特征
        :param labels: (m,1)矩阵，即m个样本的标签值
        """
        self.inputs = inputVec
        self.labels = labels
        self.dataSize = self.labels.size
        self.misclassifiedSet = np.array(np.zeros((self.dataSize, 1), dtype=int))
        # print(self.inputs.shape)
        # print(self.inputs)
        # print(self.labels.shape)
        # print(self.labels)
        # print(self.dataSize)

    def normalDistribData(self, sampleAmount, kind='AND'):
        """
        调用此成员函数，将取代默认的简化感知机，使用高斯正态分布生成模拟样本数据。
        :param sampleAmount: 样本总量
        :param kind: 感知机类型，['AND', 'OR', 'NOT']三选一，不用在意大小写
        """
        if kind.upper() not in ['AND', 'OR', 'NOT']:
            return
        sampleNum = sampleAmount // 4
        mean = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        cov = np.eye(2) * 0.01  # more bigger the coefficient more a mess
        samples = np.zeros((0, 2))  # (0,2)
        labels = np.zeros((0, 1))  # (0,1)
        for i in range(4):
            temp = np.random.multivariate_normal(mean[i], cov, sampleNum)  # 2-D normal Distribution
            samples = np.append(samples, temp, axis=0)
            if 'AND' == kind.upper():
                isPossitive = mean[i, 0] and mean[i, 1]
            elif 'OR' == kind.upper():
                isPossitive = mean[i, 0] or mean[i, 1]
            elif 'NOT' == kind.upper():
                isPossitive = mean[i, 1]  # 不考虑f1，直接取f2的值即可
            else:
                isPossitive = mean[i, 0] and mean[i, 1]
            if isPossitive:
                labels = np.append(labels, np.ones((sampleNum, 1)), axis=0)
            elif not isPossitive:
                labels = np.append(labels, np.ones((sampleNum, 1)) * -1, axis=0)
        self.getTraningDataset(samples, labels)


if __name__ == '__main__':
    p = Perceptron('not')
    print(p)
    p.normalDistribData(12, 'and')
    # p.normalDistribData(100, 'or')
    # p.normalDistribData(100, 'not')
    # p.genMisclassifiedSet()
    # p.plotBoundary()
    # p.show()
    # p.activation()
    p.training()
    print(p)
