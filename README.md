<p align="center"><img src="figures/logo_mark.png" width="104" alt="纤层智构"></p>

# 纤层智构 —— 轻量化复合材料铺层智能优化

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于经典层合板理论（CLT）与机器学习代理模型，研究 **4 层对称铺层 `[θ1/θ2/θ2/θ1]s`** 的
面内刚度 `Ex` 与弯曲刚度 `D11` 随铺层角度的变化规律，并进一步引入层数 `N` 完成
**重量–刚度双目标优化**；最后把问题推进到**真实工程结构优化**层面——以屈曲不失稳为约束、
以最轻为目标，给出单向与双向受压下的最优设计，并用理论上限证明"厚度定生死，角度定上限"，
并以单个 8 层算例与 ABAQUS 线性摄动屈曲做了数值对照（相对差异 2.94%），最终把全流程封装为带
**工程场景预设、铺层图形化与专业化 UI** 的 **Streamlit + Plotly 交互式 Web 平台**（`app.py`）。

材料常数：`E1 = 181 GPa`、`E2 = 10.3 GPa`、`G12 = 7.17 GPa`、`v12 = 0.28`、单层厚度 `0.125 mm`。

---

## 1. 目录结构

```
Final_Submission/
├── README.md                      # 本文件：环境依赖与运行顺序
├── requirements.txt               # 环境依赖清单（另有 codes/ 下同名副本）
├── 技术报告.md                     # 技术报告（含 5 张编号插图）
├── 过程日志.md                     # 完整排查过程记录（含踩坑与修复、ABAQUS 验证）
├── 过程留痕.md                     # 比赛佐证：按时间线的阶段记录（含工程化、产品化与 ABAQUS 验证阶段）
├── 实验结论_双向受压屈曲.md          # 双向受压专题结论（厚度定生死，角度定上限）
├── codes/                         # 全部源码
│   ├── test_clt.py                # ① 环境验证：单点 ABD 矩阵与工程常数
│   ├── generate_data.py           # ② LHS 采样生成数据集
│   ├── plot_data.py               # ③ 刚度云图（理论曲面 + 采样点）
│   ├── augment_data.py            # ④ 边界加密扩充数据集
│   ├── train_model.py             # ⑤ 训练随机森林代理模型
│   ├── optimize.py                # ⑥ NSGA-II 双目标优化
│   ├── structural_optimize.py     # ⑦ 单向受压屈曲约束下的最小重量设计
│   ├── angle_optimize.py          # ⑧ 双向受压铺层角度优化（GA + Ritz 屈曲）
│   ├── plot_ncr_compare.py        # ⑨ 8 层 / 12 层屈曲载荷对比图
│   ├── Auto_Design.py             # ⑩ 一键式设计工具（Nx/Ny/SF → 方案 + 屈曲/强度双重校核）
│   ├── app.py                     # ⑪ Streamlit Web 平台（滑块输入 → 铺层/3D 波形/性能对比）
│   └── requirements.txt           # 环境依赖清单
├── data/                          # 全部数据产物
│   ├── training_data.csv          # 500 组 LHS 样本
│   ├── augmented_training_data.csv# 602 组（500 LHS + 100 边界 + 2 极值）
│   ├── optimal_designs.csv        # v1 优化结果（纯角度，3 个解重合）
│   ├── optimal_designs_v2.csv     # v2 优化结果（含层数 N，真实权衡）
│   ├── model_Ex.pkl               # Ex 代理模型权重（随机森林，500 棵树）
│   ├── model_D11.pkl              # D11 代理模型权重（随机森林，500 棵树）
│   └── model_metadata.json        # 训练元数据（划分/种子/特征/R²/MSE）
├── abaqus/                        # 离线验证算例（ABAQUS 模型源文件与结果库）
│   ├── Job_1_Drone.inp / .odb     # 500/500 N/m，一阶特征值 2.1118
│   ├── Job_2_Battery.inp / .odb   # 1000/200 N/m，一阶特征值 1.9976
│   ├── Job_3_Solar.inp / .odb     # 1500/800 N/m，一阶特征值 6.5138
│   └── README_abaqus.md           # 算例说明（版本、单位制、对应关系、复现步骤）
└── figures/                       # 全部图件
    ├── data_distribution.png      # Ex / D11 随角度变化云图
    ├── pareto_front.png           # v1 帕累托前沿（退化为单点）
    ├── pareto_front_v2.png        # v2 帕累托前沿（重量 vs D11）
    ├── ncr_comparison.png         # 双向受压：纯 [0]8 vs 角度优化（含 12 层参考）
    ├── angle_modes.png            # 屈曲载荷柱状对比 + 两种铺层的屈曲模态
    ├── angle_polar.png            # 最优铺层 4 个角度的极坐标分布
    ├── structural_optimization.png# 单向受压：Ncr(N) 曲线与最优层数 N*=9
    ├── app_preview.png            # Streamlit 平台界面（推荐铺层 / 3D 波形 / 性能对比）
    └── abaqus_mode.png            # ABAQUS 一阶屈曲模态（S4R，特征值 0.60634）
```

> **关于 `abaqus/` 目录**：内含三条离线验证算例的完整 ABAQUS 模型源文件（`.inp`）、结果数据库（`.odb`）与说明文档 `README_abaqus.md`（软件版本、单位制、算例对应关系、一阶特征值与复现步骤）。评审专家可导入 `.inp` 重建模型、重现全部特征值，或直接读取 `.odb` 核对结果；未附 `.cae`，CAE 模型可由 `.inp` 重建。

### 技术报告说明

`技术报告.md` 为完整学术报告，含封面（作品名称、参赛团队编号）、约 300 字中文摘要与关键词，
正文按「背景与意义 / 理论方法与算法 / 结果与分析 / 验证与可靠性 / 结论与展望」组织；
第三章保留 3.1–3.6（3.6 为 Web 平台），第四章为验证章节（含 ABAQUS 交叉验证），
文末按 GB/T 7714 列出 6 篇参考文献并在正文上标引用。正文用标准 Markdown 图片语法引入
**代理模型权重文件（.pkl）与训练元数据已随包提供，评审专家可直接查看或验证报告中提到的精度指标。**
对应文件为 `data/model_Ex.pkl`、`data/model_D11.pkl` 与 `data/model_metadata.json`，加载后可复算 `Ex` 的 R² = 0.9968（MSE 6.08）与 `D11` 的 R² = 0.9994（MSE 0.00027）。

> 实测：按 `model_metadata.json` 记录的 `data_file`、特征组合与 `random_state = 42` 重新划分测试集，加载同一批 `.pkl` 复算，得 `Ex` R² = 0.996841、MSE = 6.07999，`D11` R² = 0.999352、MSE = 0.000268686，与元数据数值**完全一致**（差值 < 1e-9）。

`figures/` 下的五张插图（`data_distribution.png`、`pareto_front_v2.png`、
`ncr_comparison.png`、`app_preview.png`、`abaqus_mode.png`），统一编号为图 1–图 5。
公式采用 LaTeX 语法（`$...$` / `$$...$$`），建议用 Typora、VS Code 或 Pandoc 等支持
MathJax/KaTeX 的工具渲染。

若需导出 PDF，用任意 Markdown 工具渲染即可（图片为相对路径，需保持本目录结构）。
如需控制篇幅，主要手段是调整插图尺寸——图幅是页数的主导因素。

---

## 2. 环境配置

Python 版本：**3.13.2**（本项目在 Windows + Python 3.13.2 下验证通过；Streamlit Cloud 建议 3.11+）

```powershell
# 1) 创建虚拟环境（在项目根目录执行）
python -m venv .venv

# 2) 安装依赖（使用清华镜像，速度约 20~30 MB/s）
.\.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 3) 验证安装
.\.venv\Scripts\python.exe -c "import composipy, pymoo, sklearn, matplotlib; print('OK')"
```

> **国内网络提示**：若不加 `-i` 参数，从默认源下载速度仅约 100 KB/s（实测 matplotlib
> 单个包需 1 分 26 秒）。建议始终使用清华镜像，或在 `.venv\pip.ini` 中固定索引。

### 运行环境注意事项

| 事项 | 说明 |
|---|---|
| **脚本执行策略** | 本机未配置执行策略（默认 Restricted），`.ps1` 脚本会被拒绝执行。若需运行 PowerShell 脚本，请加 `-ExecutionPolicy Bypass` |
| **中文输出编码** | 本机 Python 默认 stdout 编码为 GBK(cp936)。所有脚本开头已用 `sys.stdout.reconfigure(encoding="utf-8")` 显式切换，避免中文乱码 |
| **中文字体** | 绘图脚本按字体族名查找 Microsoft YaHei / SimHei / Noto Sans CJK SC 等已安装字体（跨平台，不写死路径），找不到时自动退回英文标签（不会显示为方块） |

### 部署到 Streamlit Community Cloud

1. 把项目推到 GitHub，确保 `app.py`、`Auto_Design.py`、`angle_optimize.py` 处于可互相
   import 的位置（推荐三者同级；若沿用本包的 `codes/app.py` 布局，则入口路径填
   `codes/app.py`，并把 `requirements.txt` 放在仓库根目录）；
2. 在 Streamlit Cloud 新建应用并选择该仓库，主文件按上一条填写，Python 版本建议 3.11+；
3. 无需其他配置：应用不读写本地文件、不依赖任何绝对路径，`requirements.txt` 已包含
   `streamlit`、`plotly` 与全部计算依赖；
4. 首次打开需数秒导入依赖，之后即可用左侧滑块交互；
5. **演示模式**：云端会自动检测并跳过 GA，评委可秒开预设案例；本地运行则自动启用真实 GA。

---

## 3. 运行顺序（重要）

**两个关键约束：**

1. **所有脚本必须在本项目根目录下运行**。脚本内部使用相对路径（如 `training_data.csv`），
   读写均在当前工作目录，跨目录运行会报"找不到文件"。
2. **存在跨脚本 import 依赖**，不能单独拷贝某个脚本运行：
   - `augment_data.py` 依赖 `generate_data.py`、`plot_data.py`
   - `optimize.py` 依赖 `generate_data.py`

### 推荐执行顺序

```powershell
# ① 环境验证：算单点 ABD 矩阵与工程常数，确认 composipy 可用
python codes\test_clt.py

# ② 生成 500 组 LHS 数据  ->  training_data.csv
python codes\generate_data.py

# ③ 绘制刚度云图          ->  data_distribution.png
python codes\plot_data.py

# ④ 边界加密扩充          ->  augmented_training_data.csv（602 组）
python codes\augment_data.py

# ⑤ 训练代理模型          ->  model_Ex.pkl、model_D11.pkl、model_metadata.json
#                            （这三个文件已随包提供于 data/，无需重训即可核对指标）
python codes\train_model.py augmented_training_data.csv

# ⑥ 双目标优化（含层数 N） ->  pareto_front_v2.png、optimal_designs_v2.csv
python codes\optimize.py

# ⑦ 单向受压屈曲约束下的最小重量设计 ->  structural_optimization.png
python codes\structural_optimize.py

# ⑧ 双向受压铺层角度优化（含自检、GA、模态与极坐标图） ->  angle_modes.png、angle_polar.png
python codes\angle_optimize.py

# ⑨ 8 层 / 12 层屈曲载荷对比 ->  ncr_comparison.png
python codes\plot_ncr_compare.py

# ⑩ 一键式设计工具：输入工况直接输出推荐方案（不产生图片，只打终端报告）
python codes\Auto_Design.py 500 500 1.5

# ⑪ 启动 Streamlit Web 平台（滑块输入 Nx/Ny/SF，三列直出铺层 / 3D 波形 / 性能对比）
.\.venv\Scripts\python.exe -m streamlit run app.py
# 若按本包 codes/ 布局（在 Final_Submission 根目录执行），把 app.py 换成 codes\app.py：
# .\.venv\Scripts\python.exe -m streamlit run codes\app.py
```

> **步骤 ⑩ 是什么**：`Auto_Design.py` 是面向工程应用的**端到端工具**——输入纵向载荷 `Nx`、
> 横向载荷 `Ny`、安全系数 `SF`，自动完成"目标载荷换算 → 纯 0° 层数扫描 → 双向受压时的
> 角度 GA → 方案校验"，并直接打印一句可写进工程文件的推荐方案。
> 它支持两种用法：
>
> ```powershell
> python codes\Auto_Design.py            # 交互式输入（回车采用默认值 500 / 500 / 1.5）
> python codes\Auto_Design.py 500 500 1.5  # 命令行一次给定，便于复现
> ```
>
> 实测（`Nx = Ny = 500 N/m`、`SF = 1.5`，约 10 s）：
>
> ```
> 基于给定的载荷与安全系数，推荐采用 12 层 [-44.9/44.9/45.2/-45.2/-44.9/44.9]s 铺层，
> 实际屈曲载荷为 2129 N/m，总厚度为 1.50 mm。
> （Tsai-Hill FI = 3.2e-06）
> 【安全】屈曲与强度双重校核通过
> ```
>
> 它还会主动提示"层数定得偏保守"：若允许以角度优化替代纯 0° 基准来定层数，
> `N = 10` 层（1.25 mm）即可满足（`Ncr ≈ 1230 N/m`），可再省 2 层。

> **步骤 ⑪ 是什么**：`app.py` 是基于 Streamlit + Plotly 的**交互式生成式设计平台**，
> 带工程场景预设、铺层图形化与专业化 UI。
> 左侧滑块输入 `Nx`（0–2000）、`Ny`（0–2000）、`SF`（1.0–2.0），点击「开始优化」后，
> 右侧三列同步给出：
> ① 推荐铺层与 `Ncr`（含层数、厚度、面密度、`FI` 与均衡性）；
> ② `m = 1, n = 1` 的 3D 屈曲波形（鼠标旋转、`▶ 播放` 幅值演化，可切换 Ritz 实际模态）；
> ③ 与同层数纯 0° 的柱状对比及 `D` 矩阵对照。
>
> **界面特性（工程场景化升级）**：
> - 工程预设案例：机翼 / 电池包 / 帆板，选择后自动联动 `Nx/Ny/SF`，之后仍可手动微调；
> - 主标题下方动态副标题，按场景显示设计侧重点；
> - 铺层截面色块图：0° 红、90° 蓝、+45° 绿、−45° 黄、其他灰，悬停显示层号与角度；
> - 场景差异化校核卡片：电池包提示局部踩踏刚度，帆板提示航天级 SF ≥ 2.0；
> - 去 AI 感 UI：浅灰背景、白色悬浮卡片、Inter 字体、隐藏默认 Header/Menu/Footer；
> - 力学求解器 / 优化算法下拉框（界面展示，底层仍为 CLT + Ritz + NSGA-II）；
> - 计算状态实时反馈：`st.status` 依次显示“扫描层数 → 校核强度 → 计算完成”，替代页面变灰的等待体验；
> - 专业术语悬停帮助：求解器 / 算法下拉框均有 `help` 说明，侧边栏底部附“专业名词说明”；
> - 铺层容差说明：±1° 内的角度视为等效（如 44.9° 显示为 45° 色块），符合制造工艺误差。
>
> 启动后浏览器访问 `http://localhost:8501`；默认「快速模式」数秒出结果，
> 取消勾选则与 `Auto_Design.py` 完全一致（约 1 分钟）。
>
> 实测（`Nx = 1070`、`Ny = 870 N/m`、`SF = 1.5`）：14 层
> `[-40.3/49.8/-75.4/51.0/-40.3/49.8/-75.4]s`，`Ncr = 3469 N/m`（`λ = 3.24`），
> 较同层数纯 0° **+72.4%**，Tsai-Hill `FI = 2.96e-5`，屈曲 + 强度双重校核通过。
>
> **运行模式**：由环境变量 `APP_MODE` 明确指定，不再按操作系统猜测。
>
> | 取值 | 行为 | 耗时 |
> |---|---|---|
> | `APP_MODE=local` | 调用 `Auto_Design.py`，跑完整的层数扫描 + NSGA-II 寻优 + 强度校核 | 约 1 分钟（快速模式约 10 s） |
> | `APP_MODE=demo` | 只读取事先算好的离线预设算例，不运行 GA；未命中预设时退回纯 0° 估算并在页面标注 | 秒级 |
> | 未设置 / 取值非法 | 按 `demo` 处理，页面顶部提示实际模式 | 秒级 |
>
> 默认取 `demo`，与 Streamlit Cloud 的部署方式一致，不会在部署环境意外触发耗时优化。
>
> ```powershell
> # 本地完整计算
> $env:APP_MODE = "local"
> .\.venv\Scripts\python.exe -m streamlit run codes\app.py
> # 演示模式（读取离线预设算例）
> $env:APP_MODE = "demo"
> .\.venv\Scripts\python.exe -m streamlit run codes\app.py
> ```
>
> 注：部署在 Streamlit Cloud 的在线演示版固定使用预设案例模式；
> 如需体验完整的 NSGA-II 遗传算法实时寻优计算，请在本地以 `APP_MODE=local` 运行本项目。

> **关于 ABAQUS 参考算例**：`figures/abaqus_mode.png` 是 `[45/-45/-45/45]s` 在 ABAQUS/Standard 2024
> 中的线性摄动屈曲一阶模态（S4R 壳单元，SS1 四边简支，`Nx = Ny = 1000 N/m`）。
> 一阶特征值 `0.60634` → 606.3 N/m，与 Python 理论值 624.7 N/m 的数值差异 **2.94%**，
> 模态为 `m = n = 1` 的中心单波。**这是 v1 的单个 8 层算例，与下面三条预设不是同一模型**；
> 该早期算例为手工建模、未随包提供 `.cae`；三条预设算例的 `.inp` 与 `.odb` 已随包提供于
> `abaqus/` 目录（见 `README_abaqus.md`），早期算例的建模步骤见 `过程日志.md` 第 11 章。
>
> 应用内另有三组离线参考算例（`figures/abaqus_drone.png`、`abaqus_battery.png`、
> `abaqus_solar.png`，ODB 为 `Job_1_Drone` / `Job_2_Battery` / `Job_3_Solar`）：
>
> | 工况 (Nx / Ny) | 参考铺层 | 层数 | 一阶特征值 | 换算 Ncr | Ritz 解 | 偏差 |
> |---|---|---|---|---|---|---|
> | 500 / 500 N/m | `[-44.9/44.9/45.2/-45.2]s` | 12 | 2.1118 | 2111.8 N/m | 2128.6 N/m | +0.80% |
> | 1000 / 200 N/m | `[44.96/-45.28/-44.64/-90.0]s` | 10 | 1.9976 | 1997.6 N/m | 2008.9 N/m | +0.57% |
> | 1500 / 800 N/m | `[-45/45/45/-45]s` | 16 | 6.5138 | 6513.8 N/m | 6591.7 N/m | +1.20% |
>
> **验证限制**：这三组只比对「一阶临界载荷」。模型源文件（`.inp`）与结果库（`.odb`）已随包
> 提供于 `abaqus/`，材料、几何、边界与载荷可在 ABAQUS 中逐项复核；但未附 `.cae`，也没有
> 试验基准，因此界面只在「载荷工况命中」**且**「当前铺层与该算例参考铺层一致（角度容差 1°）」
> 时给出偏差，铺层不同则明确标注「不可比较」，数值差异不写作验证结论。

### 依赖关系与产物一览

| 步骤 | 脚本 | 读取 | 产出 |
|---|---|---|---|
| ① | `test_clt.py` | — | 终端输出（ABD 矩阵、Ex/Ey） |
| ② | `generate_data.py` | — | `training_data.csv` |
| ③ | `plot_data.py` | `training_data.csv` | `data_distribution.png` |
| ④ | `augment_data.py` | `training_data.csv` | `augmented_training_data.csv` |
| ⑤ | `train_model.py` | `augmented_training_data.csv` | `model_Ex.pkl`、`model_D11.pkl`、`model_metadata.json` |
| ⑥ | `optimize.py` | 闭式 D11（不依赖模型文件） | `pareto_front_v2.png`、`optimal_designs_v2.csv` |
| ⑦ | `structural_optimize.py` | composipy（内置 CLT） | `structural_optimization.png`、终端报告 |
| ⑧ | `angle_optimize.py` | composipy（自检用） | `angle_modes.png`、`angle_polar.png`、终端报告 |
| ⑨ | `plot_ncr_compare.py` | `angle_optimize.py`（同目录 import） | `ncr_comparison.png` |
| ⑩ | `Auto_Design.py` | `angle_optimize.py`（同目录 import） | 终端方案报告（屈曲 + Tsai-Hill 强度双重校核，无图片输出） |
| ⑪ | `app.py` | `Auto_Design.py`、`angle_optimize.py`（同目录 import） | Web 页面（终端仅打印启动信息） |

> **关于步骤 ⑦–⑨**：三个脚本自带完整自检（CLT 与 composipy 互检、Ritz 与 Navier 精确解互检、
> 级数收敛检查），运行时间分别约 1 s、25 s、2 s。三者都在 **`Final_Submission/` 根目录下**
> 以 `python codes\xxx.py` 方式运行即可：Python 会把脚本所在目录（`codes/`）加入 `sys.path`，
> 故 `plot_ncr_compare.py` 对 `angle_optimize.py` 的 `import` 成立；而图片输出路径写成
> 相对路径 `figures/`，相对的是**当前工作目录**（即包根目录）。

> **关于步骤 ⑤ 的参数**：`train_model.py` 默认读取 `training_data.csv`（500 组）。
> 若要用扩充后的数据训练，必须显式传入文件名：`python codes\train_model.py augmented_training_data.csv`。
> 本项目的最终模型即由 602 组扩充数据训练得到。
>
> **关于 `model_metadata.json`**：`train_model.py` 会把每个目标选用的特征组合写入该文件。
> 下游若要用模型预测，**必须读取该文件**而不能凭猜测配置特征——本项目 `Ex` 用 4 个特征
> （`θ1, θ2, θ1+θ2, |θ1-θ2|`）、`D11` 用 3 个特征（`θ1, θ2, θ1+θ2`），猜错会直接报特征数不匹配。
>
> **关于步骤 ⑥**：当前的 `optimize.py` 为 **v2 版本**（引入层数 `N`）。
> 它会覆盖式生成 `pareto_front_v2.png`，不会影响 v1 的产物文件。

---

## 4. 主要结论

### 4.1 铺层角度的影响规律

- `Ex` 关于 `θ1 = θ2` 对角线完全对称（因 `Ex ∝ [Q̄₁₁(θ1) + Q̄₁₁(θ2)]/4`），
  等值线呈 **L 形**：只要有一个角接近 0° 就能保持高刚度。
- `D11` 几乎只由外层角 `θ1` 决定 —— `[θ1/θ2/θ2/θ1]s` 中 `θ1` 的两层永远在最外侧，
  弯曲刚度按离中面距离的三次方加权，外层权重约为内层的 **7 倍**。
- **全局极值都在角点**：`[0/0]s` 取最大（`Ex=181.0 GPa`、`D11=1.8939 GPa·mm³`），
  `[90/90]s` 取最小（`Ex=10.3 GPa`、`D11=0.1078`）。
- `Ex` 在 `θ ≈ 52°` 附近有一个**内部局部极小值**（非单调）：因为 `Ex = A₁₁A₂₂/(h·det A)`，
  而 `det A` 中的 `A₆₆` 在 45° 之后回升。

### 4.2 代理模型精度

| 目标 | 特征组合 | 测试集 R² | MSE | RMSE |
|---|---|---|---|---|
| `Ex` | `θ1, θ2, θ1+θ2, \|θ1-θ2\|` | 0.9968 | 6.0800 | 2.4658 GPa |
| `D11` | `θ1, θ2, θ1+θ2` | 0.9994 | 0.0003 | 0.0164 |

- 特征重要性与物理规律一致：`D11` 模型中 `θ1` 占 **98%** 重要性；`Ex` 模型中 `θ1`、`θ2`
  重要性几乎相等（0.181 / 0.186），而 `θ1+θ2` 独占 0.544。
- **随机森林结构上无法外推**，在真值峰值处 `[0/0]s` 曾低估 `Ex` 达 **28.76%**。
  经边界加密后降至 **1.28%**。

### 4.3 双目标优化结果

**v1（纯角度优化）**：可行帕累托前沿**退化为单点** `[0/0]s`。
因为 `Ex` 与 `D11` 都随角度单调变差，`[0/0]s` 同时是两个目标的全局最优，
"Ex 最大 / D11 最大 / 折中"三个解在数学上是同一个点，不存在折中空间。
（`D11 ≥ 1.0` 是冗余约束，可行域内 `D11` 最小值为 1.477。）

**v2（引入层数 N ∈ [4,24]）**：得到真实的重量–刚度权衡前沿。

| 解 | N | 厚度 t (mm) | 重量 W (kg/m²) | D11 (GPa·mm³) | Ex (GPa) |
|---|---|---|---|---|---|
| 轻量化最优 | 4 | 0.500 | 0.8000 | 1.8939 | 181.0 |
| **折中解** | 9 | 1.125 | 1.8000 | 21.5723 | 181.0 |
| D11 最大 | 12 | 1.500 | 2.4000 | 51.1344 | 181.0 |

- 前沿为整数点 `N = 4,5,…,12`，完整数据见 `data/optimal_designs_v2.csv`。
- `D11 ∝ N³`、`W ∝ N`，指数差异使**增重换刚度极其划算**：
  从 `N=4` 到 `N=12`，重量仅增加 3 倍，`D11` 提升 **27 倍**。
- `Ex ≡ 181.0 GPa` 与 `N` 完全无关（单向层合板面内模量恒等于 `E1`）。
- `D11 ≥ 1.5` 在 `N ≥ 4` 时恒成立，是**冗余约束**；真正起作用的是重量上限 `N ≤ 12`。

### 4.4 屈曲约束与双向受压：厚度定生死，角度定上限

在刚度优化之上，进一步把目标提升为"**真实工程结构优化**"——以屈曲不失稳为约束、以最轻为目标：

- **单向受压**（`Nx = 1000 N/m`，`[0]N`）：`Ncr ∝ N³`，`N = 8` 时 745.6 N/m、`N = 9` 时 1061.6 N/m，
  故最优 **`N* = 9`（1.125 mm）**，比 `N = 12` 减重 **25%**。
- **双向受压**（`Nx = Ny = 1000 N/m`，固定 8 层 / 1.0 mm）：纯 `[0]8` 的 `D22` 仅为 `D11` 的
  **5.7%**，临界模态 m = 1 / n = 2（沿 y 向两个半波），`Ncr = 319.3 N/m`，**提前屈曲**；
  在 `[-90°, 90°]` 内加入**均衡铺层约束**（`+θ` 必须与 `-θ` 成对，`A16 = A26 = 0`）后，
  遗传算法收敛到 **`[45/-45/-45/45]s`**，`Ncr = 624.7 N/m`（**+95.6%**，达理论上限的 98.8%）。
- **理论上限**：由 `Q̄11 + Q̄22 + 2Q̄12` 的旋转不变性与 `Q̄66 ≤ 46.59 GPa`，可证任意 8 层
  （1.0 mm）铺层 **`Ncr ≤ 632.2 N/m < 1000 N/m`** —— **角度只能决定"逼近上限多少"，
  厚度决定"能不能行"**。
- **工程解**：10 层（1.25 mm）时，纯 `[0]10` 仅 623.6 N/m（差 38%），而均衡 ±45 达
  **1230 N/m**（裕度 23%），使达标厚度由 12 层降到 10 层、**减重 17%**；
  故正确路线是"**增厚 + 均衡角度设计**"的组合。
- **双重校核**：`Auto_Design.py` 在屈曲之外内置 **Tsai-Hill 面内强度校核**（T300/5208 典型许用值），
  `FI < 1` 打印"【安全】屈曲与强度双重校核通过"，否则提示"必须增加厚度"。
- **商用有限元复核（早期单算例）**：将 `[45/-45/-45/45]s`（8 层、1.0 mm）导入 ABAQUS
  （S4R 壳单元、SS1 四边简支、`Nx = Ny = 1000 N/m`）做线性摄动屈曲，一阶特征值
  `0.60634`（606.3 N/m），与 Python 的 624.7 N/m 相差 **2.94%**，模态为中心单波
  `m = n = 1`（见 `figures/abaqus_mode.png`）。
  ⚠️ 这是**单个模型的数值对照**，与应用内三条预设案例（500/500、1000/200、1500/800）不是
  同一个模型；应用内的对照口径见 §3「关于 ABAQUS 参考算例」——**载荷相同不等于模型相同**，
  且该早期算例未随包提供 `.cae`，只能算"条件性参考对比"，不构成验证结论。

> 专题细节见 `实验结论_双向受压屈曲.md`，代码见 `codes/angle_optimize.py`。
>
> **一键调用**：以上结论已封装为 `codes/Auto_Design.py`，输入工况即可输出推荐铺层——
> `python codes\Auto_Design.py 500 500 1.5` → 12 层
> `[-44.9/44.9/45.2/-45.2/-44.9/44.9]s`，`Ncr = 2129 N/m`，厚度 1.50 mm，
> Tsai-Hill `FI = 3.2e-6`（双重校核通过，详见 §3 步骤 ⑩）。
>
> **交互平台**：同一套计算内核已封装为 Streamlit Web 平台 `codes/app.py`，
> 滑块输入 `Nx / Ny / SF` 即可一键直出推荐铺层、3D 屈曲波形演化与性能对比
> （详见 §3 步骤 ⑪）。

---

## 5. 验证措施

所有关键结果都做了独立交叉验证，而非只信任单一实现：

- **CLT 双实现互检**：`composipy` 与从零实现的解析式 CLT 在 602 组样本上
  最大相对误差 `5.9e-10`（即 CSV 保留 10 位有效数字的舍入下限）。
- **端点解析校验**：`[0/0]s` 得 `Ex = 181.000000`（精确等于 `E1`），
  `[90/90]s` 得 `Ex = 10.300000`（精确等于 `E2`）。
- **互等关系**：`max|νxy/Ex − νyx/Ey| = 1.7e-18`（理论上为 0）。
- **标度律校验**：三个代表解的 `D11/N³` 精确等于常数 `0.0295917`。
- **模型回读校验**：`joblib` 保存后重新加载复算，R² 与训练时完全一致。
- **早期单算例有限元对照**：v1 阶段的 `[45/-45/-45/45]s`、`Nx = Ny = 1000 N/m`、`t = 1.0 mm`
  算例，ABAQUS S4R 线性摄动得到 606.3 N/m，与当时 Python 的 624.7 N/m 相差 **2.94%**，
  屈曲模态一致（`m = n = 1` 中心单波），两者都落在该模型的理论上限 632.2 N/m 附近。
  这只是**单个模型**的数值对照：层数、厚度、载荷都与 Web 应用里的三条预设案例不同，
  不构成对应用输出结果的验证结论；应用内的对照口径见第 3 章「关于 ABAQUS 参考算例」说明。

### 5.1 自动化测试（`codes/app.py`）

`Final_Submission/tests/` 下的用例不依赖浏览器，可直接用 pytest 运行：

```powershell
cd Final_Submission
..\.venv\Scripts\python.exe -m pytest tests -q
```

- `tests/test_p0_closure.py`（21 项）：不可行分支、校核结论的多状态判定、
  来源未核实的预设隔离、ABAQUS 条件性对比的三种情形。
- `tests/test_p1_inputs.py`（11 项）：载荷默认值、滑块与精确值双向同步、
  预设带入、目标载荷实时刷新、演示模式算例清单、参数改动提示、
  「开始新的设计」的状态重置与历史保留。

两个文件都以 `APP_MODE=demo` 运行，不触发 GA，全部跑完约 20 s。

---

## 6. 已知限制

1. **随机森林无法外推**：预测值必落在训练样本的目标范围内，边界极值天然被低估。
   本项目的应对是加入 102 个边界/极值样本（见 `过程日志.md` 第 4 节），
   但 `Ex` 的最大绝对误差仍达 19.99 GPa（RMSE 的 8 倍），主要来自峰值附近仍偏稀疏的采样。
2. **v1 优化问题本身退化**：纯角度优化不存在折中解，这是问题的数学性质而非算法缺陷。
3. **v2 中 `N` 为连续松弛**：NSGA-II 在 `N ∈ [4,24]` 上连续搜索，
   最终结果已投影到整数层数并用真实 CLT 复核，但前沿点由拥挤距离在目标空间铺开，
   不是逐个整数枚举得到（枚举结果已作为对照列出）。
4. **v1 的部分产物已不可复现**：当前 `optimize.py` 已是 v2 版本，
   若要重现 `pareto_front.png` / `optimal_designs.csv`，需要 v1 版本的脚本。
5. **屈曲分析采用 SS1 理想简支**：对 `D16 ≠ 0` 的弯扭耦合铺层，屈曲解对边界理想化方式
   （SS1–SS4）与级数项数有约 1%–3% 的敏感性；未计入横向剪切（`a/h ≥ 300`，薄板假设成立）、
   分层/开孔/冲击损伤与制造偏差，安全系数取 `SF = 1`（工程常用 ≥ 1.5）。
   但"8 层不可能达到 1000 N/m"的结论由与边界无关的理论上限 632.2 N/m 保证。
6. **Web 平台是单页交互工具**：不做用户管理与设计历史持久化，结果按参数缓存在进程内，
   刷新页面后需重新点击计算；云端部署时需自行配置访问权限与自定义域名。
7. **早期 v1 单算例的 ABAQUS 模型未随包提交**：该算例为手工建模，结果以截图
   `figures/abaqus_mode.png` 与 2.94% 的数值差异记录，参数见 `过程日志.md` 第 11 章；
   三条预设算例的 `.inp` 与 `.odb` 已随包提供于 `abaqus/`，可独立复现特征值。

---

## License

本项目采用 MIT License 开源协议，允许自由使用、修改和分发。

完整条文见仓库根目录的 [LICENSE](LICENSE) 文件（Copyright (c) 2026 YE-y-e）。
