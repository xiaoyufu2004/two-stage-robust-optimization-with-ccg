# **鲁棒优化笔记：静态鲁棒与两阶段鲁棒优化**
> **参考**
> Zeng B., Zhao L. (2013) *Solving two-stage robust optimization problems using a column-and-constraint generation method*, **ORL** 41 (5) .  
> 公众号文章《鲁棒优化| C&CG算法求解两阶段鲁棒优化：全网最完整、最详细的【入门-完整推导-代码实现】笔记》

## 1. 静态鲁棒优化

### 1.1 原始问题
$$\max_{x} \min_{u \in U} c^T x$$
$$\text{s.t. } Ax \geq b - Bu, \quad x \geq 0, \ u \in U$$

可以通过KKT条件转化为单层优化问题，或者用强对偶定理处理。

### **1.2 KKT条件**

#### 步骤1：固定外层变量 $u$ ，对内层问题用KKT条件
内层问题：
$$\min_{x} c^T x$$
$$\text{s.t. } Ax \geq b - Bu, \quad x \geq 0$$

#### 步骤2：引入拉格朗日乘子，构造拉格朗日函数
$$L(x,y,w) = c^T x - y^T (Ax - (b - Bu)) - w^T x$$
其中：
- $y \geq 0$：约束$Ax \geq b - Bu$的乘子
- $w \geq 0$：约束$x \geq 0$的乘子

#### 步骤3：列出KKT条件及原问题约束
1. **原始可行**：
   $$Ax \geq b - Bu, \quad x \geq 0$$

2. **拉格朗日对 $x$ 梯度为0**：
   $$\nabla_x L = -c - A^T y - w = 0 \Rightarrow A^T y + w = -c$$

   也可以从对偶问题的角度理解该条件，对偶问题：
   $$\max (b-Bu)y$$
   $$\text{s.t. }A^T y \leq c, \quad y \geq 0$$
   对于对偶问题的不等式$A^T y \leq c$，引入松弛变量$w \geq 0$化为等式$A^T y + w \leq c$，与拉格朗日函数对$x$梯度为0的条件等价。

3. **拉格朗日乘子非负**
   $$y \geq 0, \ w \geq 0$$

4. **互补松弛条件**：
   $$y^T (Ax - (b - Bu)) = 0$$
   $$w^T x = 0$$

#### 步骤4：将原问题内层替换为KKT条件，得到单层优化问题
$$\max_{x,y,w,u} c^T x$$
$$\begin{align*}
\text{s.t. } & Ax \geq b - Bu \\
& x \geq 0 \\
& A^T y + w = -c \\
& y \geq 0, \ w \geq 0 \\
& y^T (Ax - (b - Bu)) = 0 \\
& w^T x = 0 \\
& u \in U
\end{align*}$$

#### 步骤5：互补松弛条件线性化（大M法）
互补松弛条件包含两个变量的乘积（ $y^T x$ 与 $w^T x$ ），为非线性项，引入二元变量 $p_i, q_j \in \{0,1\}$ 与很大的常数 $M$ 线性化：
$$\begin{cases}
0 \leq y_i \leq M p_i \\
0 \leq (Ax - (b - Bu))_i \leq M (1-p_i) \\
0 \leq w_j \leq M q_j \\
0 \leq x_j \leq M (1-q_j)
\end{cases}$$

#### MILP模型
$$\max_{x,y,w,u,p,q} c^T x$$
$$\begin{align*}
\text{s.t. } & Ax \geq b - Bu \\
& x \geq 0 \\
& A^T y + w = -c \\
& y \geq 0, \ w \geq 0 \\
& u \in U \\
& 0 \leq y_i \leq M p_i \quad \forall i \\
& 0 \leq (Ax - (b - Bu))_i \leq M (1-p_i) \quad \forall i \\
& 0 \leq w_j \leq M q_j \quad \forall j \\
& 0 \leq x_j \leq M (1-q_j) \quad \forall j \\
& p_i, q_j \in \{0,1\} \quad \forall i,j
\end{align*}$$

### 1.3 强对偶定理
#### 步骤1：替换内层问题为对偶问题
原始问题：
$$\min_{x} c^T x \quad \text{s.t.} \quad Ax \geq b - Bu, \quad x \geq 0$$
对偶问题：
$$\max_{y} (b - Bu)^T y \quad \text{s.t.} \quad A^T y \leq c, \quad y \geq 0$$

#### 步骤2：双层优化合并

$$\max_{u \in U} \max_{y} (b - Bu)^T y \quad \text{s.t.} \quad A^T y \leq c, \quad y \geq 0$$

等价于单层问题：

$$\max_{u \in U, y} (b - Bu)^T y \quad \text{s.t.} \quad A^T y \leq c, \quad y \geq 0$$

#### 步骤3：双线性项处理
合并后的目标函数：

$$(b - Bu)^T y = b^T y - u^T (B^T y)$$

其中 $u^T (B^T y)$ 是双线性项。令 $v = B^T y$，则双线性项为 $u^T v$

采用**变量分解**：
$$u = \bar{u} + \delta u(z^+-z^-)$$
即将不确定量$u$表示为名义值与偏差值，$\bar{u}$和$\delta u$均为常数，则：
$$u^T v = \bar{u}^T v + \delta u^T v^+ - \delta u^T v^-$$
最终目标函数为：
$$\max_{y, v, z} b^T y - u^T v - \delta \bar{u}^T v^+ + \delta \bar{u}^T v^-$$

**约束设计**：
- 大M法线性化：
  
$v_j^+ = z_j^+ v_j$相当于：
$$\begin{cases}
0 \leq v_j^+ - v_j \leq M(1 - z_j^+) \\
-M z_j^+ \leq v_j^+ \leq 0 
\end{cases} \quad \forall j$$
同理处理$v_j^- = z_j^- v_j$。

- 逻辑约束：
$$z_j^+ + z_j^- \leq 1 \quad (\text{二元变量互斥})$$
$$ \sum_j (z_j^+ + z_j^-)\leq \Gamma \quad (\text{预算上限})$$

#### 完整线性化模型

$$\max_{y, v, z} b^T y - u^T v - \delta u^T v^+ + \delta u^T v^-$$

$$\begin{align*}
\text{s.t. } & A^T y \leq c \\
&v = B^T y \\
& 0 \leq v_j^+ - v_j \leq M(1 - z_j^+)  \quad \forall j \\
& -M z_j^+ \leq v_j^+ \leq 0 \quad \forall j \\
& 0 \leq v_j^- - v_j \leq M(1 - z_j^-)  \quad \forall j \\
& -M z_j^- \leq v_j^- \leq 0 \quad \forall j \\
& z_j^* + z_j^* \leq 1 \quad \forall j \\
& \sum_j (z_j^+ + z_j^-)\leq \Gamma \\
& v^+, v^- \in \mathbb{R}^n \\
& y \geq 0, v \geq 0, z_j^* \in \{0,1\}
\end{align*}$$

#### 关键技巧总结
1. **对偶转化**：将min-max问题转化为单层max问题
2. **双线性分解**： $u^T v = \bar{u}^T v + \delta u^T v^+ - \delta u^T v^-$
3. **线性化**：使用大M法和二元变量 $z_j^*$ 处理非线性项

## **2. 两阶段鲁棒优化**

### 标准模型
通用的数学模型如下所示：
$$\min_{y} c^{T}y + \max_{u \in \mathcal{U}} \min_{x \in F(y, u)} b^{T}x$$
约束条件为：
$$Ay \ge d \quad \text{(1)}$$
$$Gx \ge h - Ey - Mu, \quad \forall u \in \mathcal{U} \quad \text{(2)}$$
$$y \in S_{y} \subseteq \mathbb{R}_{+}^{n}, \quad x \in S_{x} \subseteq \mathbb{R}_{+}^{m}$$

其中：
* $y$ 和 $x$ 分别是第一和第二阶段的决策变量。
* $u$ 是不确定参数，其取值范围由不确定集 $\mathcal{U}$ 描述。
* 目标函数的第一部分 $c^Ty$ 是第一阶段成本，第二部分 $\max_{u \in \mathcal{U}} \min_{x \in F(y, u)} b^{T}x$ 是在最坏情况下的第二阶段成本。
由于不确定集 $\mathcal{U}$ 通常包含大量甚至无穷多个场景，直接枚举求解是不可行的，因此需要设计专门的算法。

参考文献介绍了**Benders-dual cutting plane method**和**Column and constraint method**两种方法，将原问题分解为主问题和子问题，对子问题（min问题）的内层取对偶（max问题）的操作是一样的。

Benders分解之后将对偶子问题的内外层（均为max问题）合并，将子问题转为单层优化问题，是双线性问题，可用启发式算法/Gurobi/KKT条件求解。求解子问题的目标函数为主问题的辅助变量 $\eta$ 提供下界，即构造最优割。 $\min_{y} c^T y + \max_{u \in U} \min_{x \in F(y,u)} b^T x$ 为原问题提供上界UB， $\min_{y} c^T y + \eta$ 为原问题提供下界LB，上下界不断改进，知道LB=UB。

CCG算法则对子问题内层用**KKT条件**替代，将子问题转化为单层优化问题。求解子问题，将可能是最坏场景的不确定量对应的**决策变量和约束**加回到主问题中。

| 特性                | CCG算法                     | Benders-Dual算法          |
|---------------------|----------------------------|--------------------------|
| **变量管理**        | 每次迭代增加新变量$x^k$     | 决策变量不变             |
| **第一阶段要求**    | 支持整数变量               | 仅支持线性规划           |
| **收敛速度**        | $O(p)$                     | $O(p \times q)$          |
| **子问题类型**      | 双线性规划                 | 线性规划                 |
| **计算复杂度**      | 较低                       | 较高                     |
| **实现难度**        | 中等                       | 较低                     |
| **适用场景**        | 复杂不确定性集合           | 简单不确定性集合         |


### **列与约束生成（C&CG）算法详解**

#### 1. 主问题 (Master Problem)

主问题用于求解第一阶段变量 $y$ ，同时利用一个辅助变量 $\eta$ 来近似最坏情况下的第二阶段成本。在C&CG算法的第 $k$ 次迭代中，主问题 $MP_2$ 的形式如下：
$$MP_2: \min_{y, \eta, \{x^l\}} c^{T}y + \eta$$
约束条件为：
$$Ay \ge d$$
$$\eta \ge b^{T}x^l, \quad l=1,...,r$$
$$Ey + Gx^l \ge h - Mu_l^*, \quad l=1,...,r$$
$$y \in S_y, \quad \eta \in \mathbb{R}, \quad x^l \in S_x, \quad l=1,...,r$$
* 每次迭代，算法会从子问题中得到一个最坏场景 $u_k^*$ 。
* 随后，一组新的第二阶段变量 $x^k$ (即“列”)和与之相关的约束被添加到主问题中。这使得主问题对第二阶段成本的估计越来越精确，其目标函数值是原问题的一个下界（Lower Bound）。

#### 2. 子问题 (Subproblem)

子问题的目标是，对于主问题给出的一个固定的第一阶段解 $\bar{y}$ ，在不确定集 $\mathcal{U}$ 中找到一个能导致第二阶段成本最大的“最坏”场景 $u^*$ 。子问题 $SP_2$ 的形式是一个双层优化问题：
$$SP_2: Q(\bar{y}) = \max_{u \in \mathcal{U}} \min_{x} \{ b^{T}x : Gx \ge h - E\bar{y} - Mu, x \in S_x \}$$
这个双层结构是求解的难点。C&CG算法通过对内层的最小化问题进行对偶变换或使用KKT条件，将其转化为单层问题。

#### 3. 子问题的KKT条件转化

为了求解子问题 $SP_2$ ，我们可以利用其内层问题是一个线性规划的特性。假设强对偶性成立（例如，满足Slater's条件），我们可以用KKT（Karush-Kuhn-Tucker）条件来等价替换内层的最小化问题。

对于给定的 $y$ 和 $u$ ，内层问题为：
$$\min_{x} b^{T}x$$
约束条件：
$$h - Ey - Mu - Gx \le 0$$
$$-x \le 0$$
通过引入对偶变量 $\pi$ 和 $\lambda$ ，其KKT条件包括原始可行性、对偶可行性以及互补松弛条件。将这些KKT条件代入外层的最大化问题，子问题 $SP_2$ 就被转化为一个等价的单层优化问题：
$$\max \quad b^{T}x$$
约束条件：
$$Gx \ge h - Ey - Mu \quad \text{(原始可行性)}$$
$$G^{T}\pi \le b \quad \text{(对偶可行性)}$$
$$(Gx - h + Ey + Mu)_i \cdot \pi_i = 0, \quad \forall i \quad \text{(互补松弛)}$$
$$(b - G^{T}\pi)_j \cdot x_j = 0, \quad \forall j \quad \text{(互补松弛)}$$
$$u \in \mathcal{U}, \quad x \in S_x, \quad \pi \ge 0$$

#### 4. 等价线性化

转化后的子问题中包含了非线性的互补松弛约束（例如 $(b - G^{T}\pi)_j \cdot x_j = 0$ ）。这些约束可以通过引入辅助二元变量和“大M”（Big-M）方法进行线性化 。

例如，对于约束 $(b - G^{T}\pi)_j \cdot x_j = 0$ ，我们可以引入一个二元变量 $v_j \in \{0, 1\}$ 和一个足够大的常数 $M$ ，将其替换为以下两个线性约束：
$$x_j \le M \cdot v_j$$
$$(b - G^{T}\pi)_j \le M \cdot (1 - v_j)$$
通过这种方式，整个子问题就可以被转化为一个混合整数线性规划（MILP），从而可以使用Gurobi等商业求解器进行有效求解。

#### 5. 迭代流程

1.  **初始化**:
    * 设置上下界 $LB = -\infty$, $UB = +\infty$ 。
    * 构建一个初始的主问题，可能只包含第一阶段的变量和约束，或者包含一个初始的、预估的场景。

2.  **迭代循环**: 在第 $k$ 次迭代中：
    * **求解主问题**: 求解当前的主问题（Master Problem），得到第一阶段决策 $y_k^* $ 和一个当前最优的目标值。这个目标值是原问题真实最优解的一个**下界 (Lower Bound)**。
        $$LB = c^T y_k^* + \eta_k^*$$
    * **求解子问题**: 将主问题得到的 $y_k^* $ 固定，代入子问题（Subproblem）中进行求解。子问题的目标是找到在此 $y_k^* $ 下，能使第二阶段成本最大化的“最坏”不确定性场景 $u_k^* $ 以及对应的第二阶段决策 $x_k^* $ 和目标值 $Q(y_k^*)$ 。
    * **更新上界**: 利用子问题的解来更新全局的**上界 (Upper Bound)**。上界是目前为止我们找到了一个可行解所对应的真实目标函数值。
        $$UB = \min(UB, \ c^T y_k^* + Q(y_k^*))$$
    * **检查收敛**: 判断上下界的差距是否满足收敛条件。
        $$\text{If } (UB - LB) / UB \le \epsilon \text{, then terminate.}$$
        如果满足，则当前解 $y_k^* $ 就是最优解，算法结束。

    * **生成列与约束**: 如果未收敛，则需要利用刚刚从子问题中得到的最坏场景 $u_k^* $ 来加强主问题：
        * **生成新列 (Column Generation)**: 在主问题中，创建一组全新的第二阶段决策变量，记为 $x^{k+1}$。这些变量代表了在第 $k+1$ 个被识别出的最坏场景下的应对策略。
        * **生成新约束 (Constraint Generation)**: 将与新变量 $x^{k+1}$ 和最坏场景 $u_k^* $ 相关的约束添加到主问题中。主要包括两种：
            1.  **优化性割平面 (Optimality Cut)**: 这个约束将辅助变量 $\eta$ 与新场景的第二阶段成本关联起来，确保 $\eta$ 能够正确地反映所有已知最坏场景中的最大成本。
                $$\eta \ge b^T x^{k+1}$$
            2.  **可行性约束 (Feasibility Constraints)**: 这组约束确保了对于场景 $u_k^* $ ，第一阶段决策 $y$ 和新的第二阶段决策 $x^{k+1}$ 必须是可行的。
                $$Gy^{k+1} \ge h - Ey - Mu_k^*$$


#### 6. 案例分析: 鲁棒选址-运输问题

文档中以一个经典的选址-运输问题作为案例，来演示C&CG算法的应用。

* **问题背景**：
    * 第一阶段：决定仓库的建设与否 ($y_i$) 及其容量 ($z_i$)。
    * 第二阶段：在需求明确后，安排从已建仓库到客户的运输方案 ($x_{ij}$) 。
    * 不确定性：客户的需求 $d_j$ 是不确定的，在一个预定义的多面体不确定集 $D$ 内波动。

* **原问题模型**
该问题的两阶段鲁棒优化模型如下：
$$\min_{y,z} \sum_{i} f_i y_i + \sum_{i} c_i z_i + \max_{d \in D} \min_{x} \sum_{i,j} t_{ij} x_{ij}$$
第一阶段约束包括：
$$z_i \le K_i y_i, \quad \forall i$$
第二阶段约束包括：
$$\sum_{j} x_{ij} \le z_i, \quad \forall i$$
$$\sum_{i} x_{ij} \ge d_j, \quad \forall j$$
不确定集 $D$ 的定义为：
$$D = \{d : d_j = \bar{d_j} + \xi_j g_j, \quad \sum g_j \le \Gamma, \dots \}$$

* **主问题与子问题**
- **主问题**：求解第一阶段变量 $y, z$ 和第二阶段成本的近似值 $\eta$。每次迭代，会根据子问题找到的最坏需求场景 $d^k$ 和对应的运输变量 $x^k$，添加新的列和约束。
- **子问题**：对于给定的 $y, z$，找到最坏的需求 $d \in D$，使得第二阶段的运输成本最小化的值达到最大。
    $$
    \max_{d \in D} \min_{x \ge 0} \{\sum_{i,j} t_{ij} x_{ij} \mid \sum_{j} x_{ij} \le z_i, \sum_{i} x_{ij} \ge d_j \}
    $$
