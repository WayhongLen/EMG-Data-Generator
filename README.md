# 肌电模拟数据生成器（EMG Data Generator）

生成模拟表面肌电（sEMG）数据的 Python 工具，输出 6 列 `.txt`，供 EMG-Analyzer 软件分析。内置**正常咬合**与**颞下颌关节紊乱（TMD）咬合**两种模式，并带指标自检、生理特征核对与鲁棒性扫描能力。

## 功能

- **两种咬合模式**（GUI 两按钮 / CLI 子命令）：
  - 正常咬合（健康对照）：软件指标 As-MM / As-TA / Ac / Tc 全部落在临床参考范围内
  - TMD 紊乱咬合（右侧患侧）：不对称指数显著超标，作异常对照
- **指标自检**：生成后自动计算四项软件同款指标（`compute_indices`）
- **生理特征定量核对**：静息张力比、患侧早启动、患/健幅值比、疲劳递减率
- **鲁棒性扫描**：4 类扰动 × 10 档强度 → 退化曲线（SVG）+ 失效阈值（CSV）
- **多种子统计报告**：100 随机种子的指标误差汇总 + 退化曲线数值（供论文 Results 引用真实数据）
- **指标插件化**：`register_metric()` 注册自定义指标，评估与自检自动纳入

## 数据规格

- 6 列 × 10000 行，500 Hz 采样（20 s）
- 列顺序（严格按模板）：`LMM, Channel 2, Channel 3, LTA, RMM, RTA`
- 时间结构：静息 → 3 次均匀咬合（各约 2.2 s）→ 收尾静息
- 波形：随机尖峰脉冲形态（快速上升/回落），非规则正弦波

## 软件指标算法

按咬合区间内各通道 RMS 计算（与 EMG-Analyzer 实测吻合）：

| 指标 | 公式 | 参考范围 |
|------|------|---------|
| As-MM 咬肌不对称 | (LMM−RMM)/(LMM+RMM)×100% | ±20% |
| As-TA 颞肌不对称 | (LTA−RTA)/(LTA+RTA)×100% | ±20% |
| Ac 活动指数 | 平均[(MM−TA)/(MM+TA)]×100% | ±10% |
| Tc 转矩指数 | [(LMM+LTA)−(RMM+RTA)]/总和×100% | ±10% |

## 运行方式

### 方式一：exe（无需 Python）
双击 `肌电数据生成器.exe`（打包产物，不进仓库，需自行用 PyInstaller 打包）。

### 方式二：源码
```bash
pip install numpy        # GUI 还需 tkinter
python EMG_Data_Generator.py                 # 打开图形界面
python EMG_Data_Generator.py normal          # 命令行生成正常咬合
python EMG_Data_Generator.py tmd             # 命令行生成 TMD
python EMG_Data_Generator.py scan 42 "输出目录"       # 鲁棒性扫描
python EMG_Data_Generator.py report 100 1 "输出目录"  # 多种子统计报告
```

## 目录结构

```
EMG_Data_Generator.py    # 主程序（GUI + CLI）
README_数据生成器.md      # 详细使用说明
LICENSE                  # MIT 许可证
```

## 许可证

[MIT](LICENSE) © 2026 冷纬杭
