# 微电网购电与储能调度优化

**Microgrid Dispatch Optimization** · 2026 全国大学生数学建模竞赛（C 题）

以 10 分钟为调度粒度，联合安排外网购电和电池充放电。在满足负载、储能容量和功率约束的前提下，比较日前计划、日内滚动调整和波动电价预测对购电成本的影响。

[建模报告（79 页）](docs/report/README.md) · [分时策略结果](results/dispatch/README.md) · [逐日账单](results/daily-bills/) · [Python 运行入口](code/run_latest.py) · [竞赛官方赛题与数据](https://www.mcm.edu.cn/html_cn/node/27b6e148f8113f09b0269f64a02629fb.html)

## 主要结果

| 分析 | 对照与结果 |
| --- | --- |
| 单日储能调度 | 购电费由无储能方案的 **48,052.05 元**降至 **35,126.95 元**，降低 **26.90%** |
| 日内滚动调整 | 2025 年 2–12 月，以同一模型中不启用日内更新的方案为基线，购电费由 **1,410.94 万元**降至 **1,345.39 万元**，降低 **4.65%**，紧急购电量由 **77,976.22 kWh**降至 **71,116.14 kWh** |
| 波动电价预测 | 逐月选模的 XGBoost 电价预测 RMSE 为 **0.05646 元/kWh**，较上周同期基线降低 **14.70%** |

第二至四问的正式结算期为 **334 天**，1 月用于历史初始化和相应策略校准。结果来自给定竞赛数据上的回测。电价预测误差和实际购电成本分别评估。

## 建模与实现

| 模块 | 方法 | 代码 |
| --- | --- | --- |
| 确定性日前调度 | 用 SciPy/HiGHS 线性规划求解购电、充放电和储电量，校验供需平衡、储能约束与原始–对偶间隙 | [Q1](code/reproduction/q1/scripts/) |
| 供需预测与日前计划 | Ridge 预测负载、Lasso 预测光伏出力。以历史误差构造两日情景，比较带风险储备的购电策略 | [预测](code/reproduction/q2/forecast_module/code/) · [Q2](code/reproduction/q2/code/optimization/) |
| 日内滚动调整 | 将整点光伏预报积分到 10 分钟粒度，比较 6:00、12:00、18:00 更新的八种组合，并做非参数检验 | [Q3](code/reproduction/q3/code/optimization/) |
| 波动电价下调度 | 比较上周同期、固定 XGBoost 和逐月选模的 XGBoost，分别核对预测误差及实际电价结算成本 | [Q4](code/reproduction/q4/code/) |

技术栈：Python 3.12、NumPy、pandas、SciPy、scikit-learn、XGBoost、Matplotlib、openpyxl。

## 运行与复现

1. 从[竞赛官方发布页](https://www.mcm.edu.cn/html_cn/node/27b6e148f8113f09b0269f64a02629fb.html)下载 C 题，将其中的 `附件` 文件夹完整复制到 `code/reproduction/C题/附件/`。该文件夹包含 `附件1.xlsx` 至 `附件4.xlsx`，以及保留原文件名的 `附件5/` Excel 模板。
2. 在仓库根目录安装依赖，重建输入并运行单日案例。

   ```bash
   python -m pip install -r code/requirements.txt
   python code/run_latest.py --stages inputs q1
   ```

   运行后可在 `code/reproduction/q1/results/solution.json` 查看单日费用、基线对比和约束校验，在 `code/reproduction/q1/figures/` 查看调度图。绘图使用系统中文字体，Linux 可安装 Noto Sans CJK。
3. 运行完整流程。

   ```bash
   python code/run_latest.py
   ```

   入口依次重建输入、训练供需预测、计算 Q1–Q3、训练电价模型、计算 Q4 并生成 Excel 填表数据。全年计算包含多轮预测和策略求解。原有 Excel 导出脚本需要另行配置 Node.js 的 `@oai/artifact-tool`，再运行 `node code/export_template_results.mjs`，写入 `generated/xlsx/`。现成 Excel 可直接查看。

展示版已复跑输入重建和 Q1，并用逐日账单和八方案工作簿核对正式期费用。全年结果沿用原项目交付文件。赛题原文与原始附件由竞赛官网提供。

## 结果文件

| 文件 | 内容 |
| --- | --- |
| [result1.xlsx](results/result1.xlsx) | 单日购电计划与储能充放电 |
| [四种策略的分时结果](results/dispatch/README.md) | 计划购电、调整购电、储能与紧急购电，共 10 份 Excel |
| [第三问_全年365天_八种情况.xlsx](results/第三问_全年365天_八种情况.xlsx) | 八种日内更新组合及校准结果 |
| [逐日账单](results/daily-bills/) | 2025 年 2–12 月的自然日费用、购电量和储电量 |

正式 Excel 购电表头从 00:10 到次日 00:10，逐日账单按 00:00 到 24:00 汇总。具体时间对齐规则见[结果口径](docs/result-notes.md)。
