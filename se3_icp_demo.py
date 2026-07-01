# argparse 用来读取命令行参数，例如 --iters 50。
import argparse

# dataclass 可以少写样板代码，适合用来保存优化结果。
from dataclasses import dataclass

# NumPy 是 Python 里最常用的矩阵/向量计算库。
import numpy as np


# 一个很小的数，用来表示“接近 0”。本文件里主要用 1e-8 判断小角度。
EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    """把三维向量 v 变成反对称矩阵 [v]x。

    如果 v = [x, y, z]，那么 [v]x @ w 等于 np.cross(v, w)。
    在李群推导里，叉乘经常写成矩阵乘法，这个函数就是做这件事。
    """
    # 把长度为 3 的向量拆成 x、y、z 三个标量，便于写矩阵。
    x, y, z = v

    # 返回 3x3 反对称矩阵。
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """SO(3) 指数映射：把三维旋转向量 phi 变成 3x3 旋转矩阵 R。

    phi 的方向是旋转轴，长度 theta 是旋转角度。
    工程里常用 Rodrigues 公式实现这个映射。
    """
    # theta 是旋转向量的模长，也就是旋转角。
    theta = np.linalg.norm(phi)

    # Phi 是 phi 的反对称矩阵，数学上常写成 phi^。
    Phi = skew(phi)

    # 当角度非常小时，sin(theta)/theta 这类表达式容易数值不稳定。
    # 所以这里用泰勒展开近似：R ~= I + Phi + 1/2 Phi^2。
    if theta < 1e-8:
        return np.eye(3) + Phi + 0.5 * Phi @ Phi

    # Rodrigues 公式里的第一个系数。
    a = np.sin(theta) / theta

    # Rodrigues 公式里的第二个系数。
    b = (1.0 - np.cos(theta)) / (theta * theta)

    # Rodrigues 公式：R = I + a*Phi + b*Phi^2。
    return np.eye(3) + a * Phi + b * Phi @ Phi


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) 对数映射：把 3x3 旋转矩阵 R 变回三维旋转向量 phi。"""
    # 对旋转矩阵，有 trace(R) = 1 + 2*cos(theta)。
    cos_theta = (np.trace(R) - 1.0) * 0.5

    # 浮点误差可能让 cos_theta 略微超过 [-1, 1]，clip 防止 arccos 出 NaN。
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 从 cos(theta) 求旋转角 theta。
    theta = np.arccos(cos_theta)

    # 小角度下 Log(R) 近似等于 R 的反对称部分。
    if theta < 1e-8:
        return np.array(
            [
                0.5 * (R[2, 1] - R[1, 2]),
                0.5 * (R[0, 2] - R[2, 0]),
                0.5 * (R[1, 0] - R[0, 1]),
            ]
        )

    # 一般角度下，从 R - R.T 的反对称部分恢复旋转向量。
    factor = theta / (2.0 * np.sin(theta))

    # vee 操作：把反对称矩阵重新取回三维向量。
    return factor * np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )


def left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) 左雅可比 J_l(phi)。

    SE(3) 指数映射里，平移不是简单等于 rho，而是 t = J_l(phi) @ rho。
    这是因为旋转和平移在刚体运动里耦合在一起。
    """
    # 旋转角。
    theta = np.linalg.norm(phi)

    # 旋转向量对应的反对称矩阵。
    Phi = skew(phi)

    # 小角度时使用泰勒展开，避免除以很小的 theta。
    if theta < 1e-8:
        return np.eye(3) + 0.5 * Phi + (1.0 / 6.0) * Phi @ Phi

    # theta^2 会多次用到，单独保存让公式更清晰。
    theta2 = theta * theta

    # 左雅可比公式的两个系数。
    a = (1.0 - np.cos(theta)) / theta2
    b = (theta - np.sin(theta)) / (theta2 * theta)

    # J_l(phi) = I + a*Phi + b*Phi^2。
    return np.eye(3) + a * Phi + b * Phi @ Phi


def left_jacobian_so3_inv(phi: np.ndarray) -> np.ndarray:
    """SO(3) 左雅可比的逆。

    SE(3) Log 里已知 t，需要恢复 rho，所以要用 rho = J_l(phi)^-1 @ t。
    """
    # 旋转角。
    theta = np.linalg.norm(phi)

    # 旋转向量对应的反对称矩阵。
    Phi = skew(phi)

    # 小角度时使用泰勒展开。
    if theta < 1e-8:
        return np.eye(3) - 0.5 * Phi + (1.0 / 12.0) * Phi @ Phi

    # 公式里会出现 theta/2。
    half_theta = 0.5 * theta

    # cot(theta/2) = 1 / tan(theta/2)。
    cot_half_theta = 1.0 / np.tan(half_theta)

    # theta^2。
    theta2 = theta * theta

    # J_l(phi)^-1 的闭式公式。
    return np.eye(3) - 0.5 * Phi + (1.0 - theta * cot_half_theta * 0.5) / theta2 * Phi @ Phi


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) 指数映射：把 6 维李代数 xi 变成 4x4 位姿矩阵 T。

    xi = [rho, phi]。
    rho 是平移相关的 3 维量，phi 是旋转相关的 3 维量。
    """
    # 前 3 维是 rho。
    rho = xi[:3]

    # 后 3 维是 phi。
    phi = xi[3:]

    # 先把 phi 转成旋转矩阵 R。
    R = so3_exp(phi)

    # 计算 SO(3) 左雅可比，用于把 rho 转成真正的平移 t。
    V = left_jacobian_so3(phi)

    # 创建 4x4 单位矩阵，作为 SE(3) 矩阵的初始值。
    T = np.eye(4)

    # 左上角 3x3 放旋转矩阵。
    T[:3, :3] = R

    # 右上角 3x1 放平移向量 t = V @ rho。
    T[:3, 3] = V @ rho

    # 返回完整位姿矩阵。
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) 对数映射：把 4x4 位姿矩阵 T 变回 6 维 xi = [rho, phi]。"""
    # 取出旋转部分 R。
    R = T[:3, :3]

    # 取出平移部分 t。
    t = T[:3, 3]

    # 对 R 做 SO(3) Log，得到旋转向量 phi。
    phi = so3_log(R)

    # 用左雅可比逆恢复 rho。
    rho = left_jacobian_so3_inv(phi) @ t

    # 把 rho 和 phi 拼成一个 6 维向量。
    return np.hstack([rho, phi])


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """用位姿 T 批量变换点云 points。

    每个点满足 p' = R*p + t。
    NumPy 中 points 是 N x 3，所以写成 points @ R.T + t。
    """
    # T[:3, :3] 是 R，T[:3, 3] 是 t。
    return points @ T[:3, :3].T + T[:3, 3]


@dataclass
class IcpResult:
    """保存 ICP 优化结果。"""

    # 最终估计出的 4x4 位姿矩阵。
    T: np.ndarray

    # 每次迭代的 RMSE，用来观察误差是否下降。
    rmse_history: list[float]

    # 每次迭代的更新量范数，用来判断是否收敛。
    step_history: list[float]


def estimate_point_to_point_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 30,
    damping: float = 1e-8,
    verbose: bool = True,
) -> IcpResult:
    """用 SE(3) Gauss-Newton 估计 T，使 target ~= T * source。

    这个教学版本假设对应关系已知：source[i] 对应 target[i]。
    更新方式是左扰动：T_new = exp(delta_xi^) @ T。
    """
    # ICP 要求源点和目标点一一对应，所以形状必须相同。
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")

    # 初始位姿设为单位矩阵，表示一开始猜测没有旋转也没有平移。
    T = np.eye(4)

    # 用列表保存每轮误差。
    rmse_history = []

    # 用列表保存每轮更新量大小。
    step_history = []

    # Gauss-Newton 迭代。
    for iteration in range(max_iters):
        # 用当前位姿 T 把源点云变换到目标点云坐标系。
        transformed = transform_points(T, source)

        # 残差 e = T*p - q。优化目标是让这些残差尽量接近 0。
        residuals = transformed - target

        # H 是近似 Hessian，大小 6x6，因为优化变量 delta_xi 是 6 维。
        H = np.zeros((6, 6))

        # b 是一阶项，大小 6。
        b = np.zeros(6)

        # cost 保存平方误差总和，用来计算 RMSE。
        cost = 0.0

        # 遍历每一对对应点。
        for x, e in zip(transformed, residuals):
            # 当前点变换后的坐标是 x = T*p。
            # 左扰动下：exp(delta)*x ~= x + delta_rho - [x]x delta_phi。
            # 所以残差对 delta 的雅可比是 J = [I, -[x]x]。
            J = np.hstack([np.eye(3), -skew(x)])

            # Gauss-Newton 累加 H = sum(J.T @ J)。
            H += J.T @ J

            # 累加 b = sum(J.T @ e)。后面解 delta = -H^-1 b。
            b += J.T @ e

            # 累加平方误差 e.T @ e。
            cost += e @ e

        # 给 H 的对角线加一个很小的阻尼，提升数值稳定性。
        H += damping * np.eye(6)

        # 解正规方程 H * delta = -b。
        delta = -np.linalg.solve(H, b)

        # 把 6 维更新量通过 SE(3) Exp 变成矩阵，再左乘更新当前位姿。
        T = se3_exp(delta) @ T

        # RMSE = sqrt(平均平方误差)，比 cost 更直观。
        rmse = np.sqrt(cost / len(source))

        # 更新量的模长，越接近 0 表示越接近收敛。
        step_norm = np.linalg.norm(delta)

        # 记录本轮 RMSE。
        rmse_history.append(rmse)

        # 记录本轮步长。
        step_history.append(step_norm)

        # 如果 verbose=True，就打印每一轮的优化状态。
        if verbose:
            print(
                f"iter {iteration:02d} | rmse={rmse:.8f} | "
                f"|delta|={step_norm:.3e} | delta={np.array2string(delta, precision=3)}"
            )

        # 如果更新量已经非常小，就认为优化收敛，提前退出。
        if step_norm < 1e-10:
            break

    # 把最终位姿和历史信息打包返回。
    return IcpResult(T=T, rmse_history=rmse_history, step_history=step_history)


def make_demo_data(seed: int = 7, n: int = 80, noise_sigma: float = 0.003):
    """生成一组可复现实验数据。

    source 是随机点云。
    target 是把 source 用真实位姿 T_true 变换后，再加一点高斯噪声得到的。
    """
    # 创建随机数生成器；固定 seed 可以让每次运行结果一致。
    rng = np.random.default_rng(seed)

    # 生成 n 个三维点，每个坐标在 [-1, 1]。
    source = rng.uniform(-1.0, 1.0, size=(n, 3))

    # 让点云不是完全平面的，避免几何退化。
    source[:, 2] += 0.4 * source[:, 0] ** 2 - 0.2 * source[:, 1]

    # 真实位姿的李代数表示：[rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]。
    true_xi = np.array([0.35, -0.20, 0.45, 0.25, -0.18, 0.12])

    # 把真实 6 维扰动转成 4x4 位姿矩阵。
    T_true = se3_exp(true_xi)

    # 用真实位姿变换源点云，得到理想目标点云。
    target = transform_points(T_true, source)

    # 给目标点云加入高斯噪声，模拟传感器测量误差。
    target += rng.normal(scale=noise_sigma, size=target.shape)

    # 返回源点、目标点和真实位姿，后面用真实位姿检查估计误差。
    return source, target, T_true


def print_pose(name: str, T: np.ndarray) -> None:
    """打印位姿矩阵和它的李代数形式。"""
    # 把 4x4 位姿矩阵转成 6 维向量，便于观察误差。
    xi = se3_log(T)

    # 打印标题。
    print(f"\n{name}")

    # 打印矩阵本身。
    print(np.array2string(T, precision=5, suppress_small=True))

    # 打印 Log(T)，也就是 [rho, phi]。
    print(f"log(T) = [rho, phi] = {np.array2string(xi, precision=5)}")


def main() -> None:
    """程序入口：读取参数、生成数据、运行 ICP、打印结果。"""
    # 创建命令行参数解析器。
    parser = argparse.ArgumentParser(
        description="Tiny SE(3) point-to-point ICP demo with Gauss-Newton."
    )

    # 最大迭代次数。
    parser.add_argument("--iters", type=int, default=30)

    # 生成多少个点。
    parser.add_argument("--points", type=int, default=80)

    # 给目标点云加多大的噪声。
    parser.add_argument("--noise", type=float, default=0.003)

    # 随机种子。
    parser.add_argument("--seed", type=int, default=7)

    # 真正解析命令行参数。
    args = parser.parse_args()

    # 根据参数生成演示数据。
    source, target, T_true = make_demo_data(
        seed=args.seed,
        n=args.points,
        noise_sigma=args.noise,
    )

    # 提醒用户：这里使用的是已知对应关系的点到点配准。
    print("Solving target ~= T * source with known correspondences.")

    # 运行 SE(3) ICP 优化。
    result = estimate_point_to_point_icp(source, target, max_iters=args.iters)

    # 打印真实位姿。
    print_pose("True pose T_true", T_true)

    # 打印估计位姿。
    print_pose("Estimated pose T_est", result.T)

    # 计算相对误差：如果估计完全正确，inv(T_true) @ T_est 应该接近单位矩阵。
    T_error = np.linalg.inv(T_true) @ result.T

    # 把相对误差转成 6 维向量。
    err = se3_log(T_error)

    # 打印相对误差矩阵。
    print_pose("Relative error inv(T_true) @ T_est", T_error)

    # 分别打印平移误差范数和旋转误差范数。
    print(
        f"\nFinal translation/rotation error norm: "
        f"{np.linalg.norm(err[:3]):.6e}, {np.linalg.norm(err[3:]):.6e}"
    )


# Python 文件被直接运行时，__name__ 会等于 "__main__"。
# 这样写可以避免这个文件被 import 时自动执行 main()。
if __name__ == "__main__":
    main()
