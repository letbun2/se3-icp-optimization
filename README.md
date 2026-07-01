# se3-icp-optimization
A minimalistic Python implementation of SO(3)/SE(3) Lie group operations and Gauss-Newton point-to-point ICP algorithm using NumPy.

这是一个极其精简且易于理解的 3D 计算机视觉/机器人位姿估计教学示例。项目在不依赖任何复杂优化库（如 Ceres、GTSAM）的情况下，纯手工使用 **NumPy** 实现了三维刚体运动的李群李代数底层数学，并应用左扰动高斯-牛顿法完成了已知对应关系的点到点 ICP（Iterative Closest Point）配准。

## 核心特性 (Key Features)

- **纯 NumPy 实现 (Pure NumPy Implementation)**：零额外外部依赖，代码逻辑清晰，适合作为 SLAM 进阶与几何深度学习的基础练习。
- **完整的李群李代数映射 (Lie Group & Lie Algebra Operations)**：
  - $\text{SO}(3)$ 与 $\text{SE}(3)$ 之间的指数映射 (`exp`) 与对数映射 (`log`)。
  - 使用罗德里格斯公式（Rodrigues' formula）及针对小角度数值稳定的泰勒展开。
  - **$\text{SO}(3)$ 左雅可比矩阵 (`left_jacobian_so3`)** 及其逆矩阵的完整闭式解实现（严格推导平移向量与李代数之间的耦合关系）。
- **左扰动位姿优化 (Left-Perturbation Optimization)**：
  - 基于高斯-牛顿法（Gauss-Newton）直接在流形上进行迭代求解。
  - 精确实现了解析雅可比矩阵的构建与信息矩阵（Hessian）的累加。
- **可复现的仿真闭环 (Reproducible Simulation)**：内置带有高斯噪声的随机点云生成器，并提供与真实位姿（Ground Truth）对比的相对误差分析。

## 数学背景 (Mathematical Background)

本实现遵循主流 SLAM（如《视觉SLAM十四讲》）中的经典数学定义。

### 1. 反对称矩阵 (Skew-symmetric matrix)

对于三维向量 $\mathbf{v}$，其反对称矩阵表示为：

$$
\mathbf{v}^\wedge
$$

### 2. SE(3) 指数映射 (Exponential Mapping)

李代数 $\boldsymbol{\xi} = [\boldsymbol{\rho}, \boldsymbol{\phi}]^T$ 到李群 $\text{SE}(3)$ 的指数映射公式为（采用分块矩阵形式）：

$$
\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix} \exp(\boldsymbol{\phi}^\wedge) & \mathbf{J}_l \boldsymbol{\rho} \\\\ \mathbf{0}^T & 1 \end{bmatrix}
$$

### 3. 误差状态更新 (Left-Perturbation)

采用左扰动模型，每次迭代的位姿更新方式为：

$$
\mathbf{T} \leftarrow \exp(\delta\boldsymbol{\xi}^\wedge) \cdot \mathbf{T}
$$

### 4. 解析雅可比矩阵 (Jacobian)

观测点对扰动的解析雅可比矩阵为：

$$
\mathbf{J} = \begin{bmatrix} \mathbf{I}_3 & -[\mathbf{x}]_\times \end{bmatrix}
$$

## 依赖项 (Dependencies)

- Python 3.x
- NumPy

## 使用方法 (Usage)

你可以直接运行脚本来观察 ICP 的收敛过程以及最终与真实位姿的残差比对：

```bash
python main.py --iters 30 --points 80 --noise 0.003
