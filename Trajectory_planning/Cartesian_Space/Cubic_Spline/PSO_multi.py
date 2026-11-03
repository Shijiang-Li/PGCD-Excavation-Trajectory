import numpy as np


class PSO:
    def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=2, c2=2):
        """
        参数初始化
        :param func: 目标函数
        :param dim: 搜索维度
        :param pop: 种群数量
        :param max_iter: 迭代次数
        :param lb: 搜索下界
        :param ub: 搜索上界
        :param w: 惯性权重
        :param c1: 个体学习因子
        :param c2: 社会学习因子
        """
        self.func = func
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pop = pop  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)  # 下界
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)  # 上界

        # 初始化种群位置和速度
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        self.v_max = 0.1 * (self.ub - self.lb)  # 例如 v_max = 0.1 * (ub - lb)
        self.V = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop, self.dim))

        # 初始化个体最优位置和适应度
        self.pbest = self.X.copy()
        self.pbest_fitness = np.array([float('inf')] * self.pop)

        # 初始化全局最优
        self.gbest = np.zeros(self.dim)
        self.gbest_fitness = float('inf')

        # 用于记录每次迭代的最优值
        self.history = []

        # 添加一个计数器，跟踪连续无效解的迭代次数
        self.invalid_count = 0

    def update_fitness(self):
        """
        批量计算整个种群的适应度
        """
        # 同时计算所有粒子的适应度值
        # fitness = np.array([self.func(x) for x in self.X])

        fitness = self.func(self.X)

        # 检查是否所有解都是无效的
        if np.all(np.isinf(fitness)):
            self.invalid_count += 1
            # print(f"第{self.invalid_count}次迭代所有粒子无效")
            self._reset_some_particles()  # 重新初始化
            # 重新计算适应度
            fitness = self.func(self.X)
        else:
            self.invalid_count = 0

        # print(self.X, fitness)

        # 更新个体最优
        update_idx = fitness < self.pbest_fitness
        self.pbest[update_idx] = self.X[update_idx]
        self.pbest_fitness[update_idx] = fitness[update_idx]

        # 更新全局最优
        if np.min(fitness) < self.gbest_fitness:
            self.gbest = self.X[np.argmin(fitness)].copy()
            self.gbest_fitness = np.min(fitness)

        # 记录当前迭代的最优值
        self.history.append(self.gbest_fitness)

    def _reset_some_particles(self,):
        # 初始化种群位置和速度
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        self.v_max = 0.1 * (self.ub - self.lb)  # 例如 v_max = 0.1 * (ub - lb)
        self.V = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop, self.dim))

    def update_position(self):
        """
        批量更新整个种群的位置和速度
        """
        # 更新速度
        r1 = np.random.random((self.pop, self.dim))
        r2 = np.random.random((self.pop, self.dim))

        self.V = self.w * self.V + \
                 self.c1 * r1 * (self.pbest - self.X) + \
                 self.c2 * r2 * (self.gbest - self.X)

        # 更新位置
        self.X = self.X + self.V

        # 边界处理
        self.X = np.clip(self.X, self.lb, self.ub)

    def run(self):
        """
        运行PSO算法
        """
        self.history = []  # 清空历史记录
        for iter_ in range(self.max_iter):
            self.update_fitness()
            self.update_position()

        return self.gbest, self.gbest_fitness, self.history
