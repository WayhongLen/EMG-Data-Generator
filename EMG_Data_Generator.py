# -*- coding: utf-8 -*-
"""
肌电信号模拟数据生成器 v4
==========================
功能：生成模拟 sEMG 数据（.txt），供 EMG-Analyzer 软件分析。
两种模式：
  1. 正常咬合模式（健康对照）——软件指标 As-TA / As-MM / Ac / Tc 全部落在参考范围内
  2. 颞下颌关节紊乱（TMD）咬合模式（右侧为患侧）——不对称指数显著超标，作异常对照

v4 新增（鲁棒性验证体系）：
  - 扰动注入：单侧加性噪声 / 单侧削顶饱和 / 单侧电极串扰 / 通道增益不一致
  - 鲁棒性扫描：固定种子遍历扰动强度 → 指标退化曲线（SVG）+ 失效阈值（CSV）
  - 失效阈值定义：任一指标平均绝对值首次越出临床参考带对应的扰动强度
  - GUI 新增「鲁棒性扫描」按钮；CLI 支持 scan 子命令

v3 新增（平台化升级）：
  - 指标插件化：register_metric() 注册自定义指标，评估/自检自动纳入
  - 生理特征定量核对：静息张力比、早启动、患/健幅值比、疲劳递减率
  - 完成对话框输出「指标 + 生理特征」双层核对

v2 关键改进（对齐软件指标算法）：
  - 软件指标公式（按咬合区间内各通道 RMS 计算）：
      As-TA = (LTA-RTA)/(LTA+RTA)*100     参考范围 ±20%
      As-MM = (LMM-RMM)/(LMM+RMM)*100     参考范围 ±20%
      Ac    = 平均[(MM-TA)/(MM+TA)]*100   参考范围 ±10%
      Tc    = [(LMM+LTA)-(RMM+RTA)]/总和*100  参考范围 ±10%
  - 时间结构模仿模板 HT.txt：静息 -> 3 次均匀咬合（各约 2.2s）-> 收尾静息
  - 每个咬合区间按目标 RMS 闭环校准：双侧对称（|As|<6%）、咬肌/颞肌比例健康（Ac 约 -6%）
  - 漂移全部为双极性对称（不再使用单侧大漂移，避免 As 指数超标）
  - 幅值量级对齐模板（咬合区间 RMS 约 20~35，波形峰值 ±100~200）

规格：
  - 6 列、10000 行、500 Hz 采样
  - 列顺序（严格按模板）：LMM, Channel 2, Channel 3, LTA, RMM, RTA
  - Channel 2 / Channel 3 为基线列（直流偏置，量级与模板一致）

输出：与脚本/exe 同目录生成 <模式>_<时间戳>.txt
"""

import os
import sys
from datetime import datetime

import numpy as np

# tkinter 为可选依赖：无 GUI 环境下仍可调用 generate() 生成数据
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _HAS_TK = True
except ImportError:
    _HAS_TK = False


# ============ 全局参数 ============
N_ROWS = 10000          # 总行数
FS = 500                # 采样频率 Hz

# 时间结构（模仿模板：静息 -> 咬合1 -> 静息 -> 咬合2 -> 静息 -> 咬合3 -> 收尾）
BITE_INTERVALS = [(1500, 3800), (4400, 6700), (7300, 9600)]   # 3 次咬合区间
REST_PARTS = [(0, 1500), (3800, 4400), (6700, 7300), (9600, 10000)]

# 列索引
COL = {'LMM': 0, 'CH2': 1, 'CH3': 2, 'LTA': 3, 'RMM': 4, 'RTA': 5}

# 随机数生成器（可复现）
_rng = np.random.default_rng()


def set_seed(seed=None):
    """设置随机种子，seed 为 None 时使用时间熵。"""
    global _rng
    _rng = np.random.default_rng(seed)


# ============ 波形工具 ============
def emg_burst(rng, n, base_amp, rise=6, fall=10):
    """
    单个肌电爆发（尖峰脉冲簇）：快速上升、快速回落，非规则正弦波。
    绝对峰值校准到 base_amp（正负大致对称）。
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n)
    # 包络：快速上升 + 指数回落
    envelope = np.exp(-t / fall) * (1.0 - np.exp(-t / rise))
    env_max = envelope.max()
    if env_max > 0:
        envelope = envelope / env_max
    # 载波：零均值高频噪声 + 稀疏尖峰
    carrier = rng.normal(0, 1, n)
    spike_mask = rng.random(n) < 0.3
    carrier = carrier + np.where(spike_mask, rng.normal(0, 2.0, n), 0.0)
    burst = envelope * carrier
    pk = np.max(np.abs(burst))
    if pk > 0:
        burst = burst / pk * base_amp
    return burst


def emg_active_segment(rng, n, irregular=False, onset_immediate=False):
    """
    生成一段密集爆发的肌电活动信号（RMS 归一化到 1）。
    形态：随机尖峰脉冲簇连续成簇 + 持续活动基底，与模板波形形态一致。
    irregular=True 时节律不规则（TMD 特征：每次收缩幅值、时长差异大）。
    onset_immediate=True 时第一个 burst 从位置 0 开始（确定性早启动，用于患侧早接触）。
    """
    n = int(n)
    sig = np.zeros(n)
    pos = 0
    first = True
    while pos < n:
        if irregular:
            blen = int(rng.uniform(50, 350))   # 收缩时长差异大
            gap = int(rng.uniform(10, 110))    # 节律不规则
            amp_j = rng.uniform(0.5, 1.3)      # 幅值差异大
        else:
            blen = int(rng.uniform(90, 280))
            gap = int(rng.uniform(12, 45))
            amp_j = rng.uniform(0.8, 1.15)
        if first and onset_immediate:
            blen = int(rng.uniform(100, 200))  # 首个 burst 确定性起跳
            gap = 0
            amp_j = 1.0
            first = False
        blen = min(blen, n - pos)
        if blen > 0:
            sig[pos:pos + blen] += emg_burst(rng, blen, 1.0) * amp_j
        pos += blen + gap
    # 持续低幅活动基底（间隙里也有小活动）
    sig += rng.normal(0, 0.25, n)
    r = np.sqrt(np.mean(sig ** 2))
    if r > 0:
        sig = sig / r
    return sig


def add_bipolar_drift(rng, n, amp):
    """
    双极性低频漂移（正负对称摆动，均值≈0）。
    对 RMS 贡献对称，不会造成左右/单侧 RMS 失衡。
    """
    t = np.arange(n) / FS
    d = np.sin(2 * np.pi * 0.12 * t + rng.uniform(0, 2 * np.pi))
    d += 0.5 * np.sin(2 * np.pi * 0.27 * t + rng.uniform(0, 2 * np.pi))
    d += 0.3 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))
    mx = np.max(np.abs(d))
    if mx > 0:
        d = d / mx
    return d * amp


def tremor_sig(rng, n, amp, f=12.0):
    """细微震颤样高频波动（患侧特征）。"""
    t = np.arange(n) / FS
    return amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))


def rest_base(rng, n, sigma, lowfreq_amp=1.2):
    """静息基底：小随机噪声 + 极低频微波动。"""
    sig = rng.normal(0, sigma, n)
    t = np.arange(n) / FS
    sig += lowfreq_amp * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi))
    sig += 0.6 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 2 * np.pi))
    return sig


def calibrate_rms(sig, target):
    """将信号 RMS 缩放到目标值。"""
    cur = np.sqrt(np.mean(sig ** 2))
    if cur > 0:
        return sig * (target / cur)
    return sig


# ============ 指标注册机制（插件化，可扩展第三方指标） ============
# 指标 = (名称, 计算函数, 参考范围文本, 参考上限)
#   计算函数签名：fn(data, a, b) -> float，按单个咬合区间 [a,b) 计算，返回百分数
# 自定义指标：调用 register_metric(name, fn, ref, limit) 即可加入评估与自检。

_METRICS = []   # 按注册顺序排列


def _rms(data, a, b, ch):
    """区间 [a,b) 内通道 ch 的 RMS。"""
    return float(np.sqrt(np.mean(data[a:b, COL[ch]] ** 2)))


def _metric_as_mm(data, a, b):
    return (_rms(data, a, b, 'LMM') - _rms(data, a, b, 'RMM')) / \
           (_rms(data, a, b, 'LMM') + _rms(data, a, b, 'RMM')) * 100


def _metric_as_ta(data, a, b):
    return (_rms(data, a, b, 'LTA') - _rms(data, a, b, 'RTA')) / \
           (_rms(data, a, b, 'LTA') + _rms(data, a, b, 'RTA')) * 100


def _metric_ac(data, a, b):
    ac_l = (_rms(data, a, b, 'LMM') - _rms(data, a, b, 'LTA')) / \
           (_rms(data, a, b, 'LMM') + _rms(data, a, b, 'LTA')) * 100
    ac_r = (_rms(data, a, b, 'RMM') - _rms(data, a, b, 'RTA')) / \
           (_rms(data, a, b, 'RMM') + _rms(data, a, b, 'RTA')) * 100
    return (ac_l + ac_r) / 2


def _metric_tc(data, a, b):
    left = _rms(data, a, b, 'LMM') + _rms(data, a, b, 'LTA')
    right = _rms(data, a, b, 'RMM') + _rms(data, a, b, 'RTA')
    return (left - right) / (left + right) * 100


def register_metric(name, fn, ref, limit):
    """注册一个自定义指标。fn(data, a, b)->float（百分数）。"""
    _METRICS.append((name, fn, ref, limit))


def _register_defaults():
    register_metric('As-MM 咬肌不对称', _metric_as_mm, '±20%', 20.0)
    register_metric('As-TA 颞肌不对称', _metric_as_ta, '±20%', 20.0)
    register_metric('Ac 活动指数', _metric_ac, '±10%', 10.0)
    register_metric('Tc 转矩指数', _metric_tc, '±10%', 10.0)


_register_defaults()


def compute_indices(data, intervals=None):
    """
    按咬合区间计算全部已注册指标。
    返回 [(m1, m2, ...), ...] 每个区间一组（百分数），列顺序同 _METRICS。
    """
    if intervals is None:
        intervals = BITE_INTERVALS
    out = []
    for (a, b) in intervals:
        out.append(tuple(fn(data, a, b) for (_n, fn, _r, _l) in _METRICS))
    return out


def format_indices(indices):
    """把指标列表格式化为可读文本（含平均值与判定）。"""
    arr = np.array(indices)   # shape (n, n_metrics)
    lines = []
    all_ok = True
    means = arr.mean(axis=0)
    for j, (nm, _fn, rf, lim) in enumerate(_METRICS):
        per = "  ".join(f"{v:+.1f}%" for v in arr[:, j])
        avg = means[j]
        ok = abs(avg) < lim
        all_ok = all_ok and ok
        mark = "✓" if ok else "✗ 超范围"
        lines.append(f"  {nm}（参考 {rf}）：{per}  | 平均 {avg:+.1f}% {mark}")
    header = "咬合区间指标：" if not all_ok else ""
    tail = "\n  → 全部指标处于参考范围内 ✓" if all_ok else ""
    return header + "\n".join(lines) + tail


# ============ 生理特征定量核对（第三层验证：波形/生理特征可验证） ============
def compute_physio_features(data):
    """
    定量核对生成数据的生理特征，输出可读文本。
    用于把「内置生理特征」升级为「可验证特征」。
    """
    lines = []

    # 1. 静息期患/健 RMS 比（静息张力）
    #    用咬合1 之前的纯静息段 [0, a1-120)，留出患侧早启动(lag≤100点)的余量，避开能量泄漏
    a1, b1 = BITE_INTERVALS[0]
    pure_rest_end = max(a1 - 120, 200)
    pure_rest = (0, pure_rest_end)

    def seg_rms(ch, parts):
        if isinstance(parts, tuple):
            parts = [parts]
        vals = [data[a:b, COL[ch]] for (a, b) in parts]
        return float(np.sqrt(np.mean(np.concatenate(vals) ** 2)))

    lmm_rest = seg_rms('LMM', pure_rest)
    rmm_rest = seg_rms('RMM', pure_rest)
    lta_rest = seg_rms('LTA', pure_rest)
    rta_rest = seg_rms('RTA', pure_rest)
    lines.append(f"  静息 RMS（咬合前纯静息段）：LMM={lmm_rest:.2f}  RMM={rmm_rest:.2f}  "
                 f"LTA={lta_rest:.2f}  RTA={rta_rest:.2f}")
    lines.append(f"  静息患/健 RMS 比（咬肌）= {rmm_rest / max(lmm_rest, 1e-6):.2f}  "
                 f"（颞肌）= {rta_rest / max(lta_rest, 1e-6):.2f}")

    # 2. 早启动（患侧咬合1 起点能量提前量）
    #    在咬合1 前缘 [a1-250, a1+50] 窗口内，找患/健能量首次超过静息阈值的采样点，
    #    比较先后 → 量化患侧早启动
    probe = 250
    s0 = max(a1 - probe, 0)
    win = data[s0:a1 + 50]
    thr_l = float(lmm_rest * 2.5)
    thr_r = float(rmm_rest * 2.5)
    WIN_W = 20

    def onset_point(vals, thr):
        # 用滑动窗口 RMS 找首次超过阈值的点
        r = np.sqrt(np.convolve(vals ** 2, np.ones(WIN_W) / WIN_W, mode='valid'))
        idx = np.argmax(r > thr)
        return idx if r[idx] > thr else len(r)

    onset_l = onset_point(win[:, COL['LMM']], thr_l)
    onset_r = onset_point(win[:, COL['RMM']], thr_r)
    lead_ms = int((onset_l - onset_r) / FS * 1000)
    if onset_r < onset_l and lead_ms >= 20:
        lines.append(f"  早启动：患侧 RMM 较健侧 LMM 提前 {lead_ms}ms 起跳")
    else:
        lines.append(f"  早启动：双侧同步起跳（患侧 {onset_r} 点，健侧 {onset_l} 点）")

    # 3. 患/健 幅值比（咬合2 持续段，RMS 口径）
    a2, b2 = BITE_INTERVALS[1]
    ratio_mm = _rms(data, a2, b2, 'RMM') / max(_rms(data, a2, b2, 'LMM'), 1e-6)
    ratio_ta = _rms(data, a2, b2, 'RTA') / max(_rms(data, a2, b2, 'LTA'), 1e-6)
    lines.append(f"  持续咬合 患/健 RMS 比：咬肌={ratio_mm:.2f}  颞肌={ratio_ta:.2f}")

    # 4. 疲劳递减率（咬合3 相对咬合2 的 RMS 变化）
    a3, b3 = BITE_INTERVALS[2]
    for ch, nm in [('RMM', 'RMM 患侧'), ('LMM', 'LMM 健侧')]:
        r2 = _rms(data, a2, b2, ch)
        r3 = _rms(data, a3, b3, ch)
        drop = (1 - r3 / max(r2, 1e-6)) * 100
        lines.append(f"  疲劳递减率（{nm}）= {drop:.1f}%")

    return "\n".join(lines)


# ============ 通道构建 ============
def build_channel(rng, name, targets, rest_sigma, lag=0, fatigue=None,
                  drift_amp=0.0, irregular=False, tremor_amp=0.0,
                  onset_immediate=False):
    """
    构建单个肌电通道。
    targets: 每个咬合区间的目标 RMS（列表，长度 3）
    fatigue: 咬合3 的疲劳形态系数 (start, end)，None 表示不疲劳
    drift_amp: 咬合3 区间内叠加的双极性漂移幅值
    lag: 激活提前量（采样点，>0 表示整体提前启动，模拟患侧早接触）
    onset_immediate: 咬合段首个 burst 确定性起跳（配合 lag 保证患侧早启动可检出）
    """
    sig = rest_base(rng, N_ROWS, rest_sigma)
    if tremor_amp > 0:
        sig += tremor_sig(rng, N_ROWS, tremor_amp)

    for i, (a, b) in enumerate(BITE_INTERVALS):
        seg = emg_active_segment(rng, b - a, irregular=irregular,
                                 onset_immediate=onset_immediate)
        # 疲劳形态（幅值沿区间递减）
        if fatigue is not None and i == 2:
            fat = np.linspace(fatigue[0], fatigue[1], b - a)
            seg = seg * fat
            # 静默处理交给外层（通过在 seg 上置零实现）
        # 双极性漂移（咬合3 段内，正负对称不破坏指标）
        if drift_amp > 0 and i == 2:
            seg = seg + add_bipolar_drift(rng, b - a, drift_amp)
        # 校准到目标 RMS（校准保证指标可控）
        seg = calibrate_rms(seg, targets[i])
        # 放置（患侧整体左移 lag 点 → 早启动）
        if lag > 0:
            place = np.zeros(N_ROWS)
            lo = max(a - lag, 0)
            hi = min(b - lag, N_ROWS)
            place[lo:hi] = seg[:hi - lo]
            sig += place
        else:
            sig[a:b] += seg
    return sig


# ============ 正常咬合模式 ============
def generate_normal(rng):
    """
    健康正常咬合模式：双侧对称、咬肌/颞肌比例健康。
    目标 RMS（每区间）：咬肌约 26/25.3，颞肌约 29/29.6（与模板量级一致）
    指标预期：As-MM ≈ +1%，As-TA ≈ -1%，Ac ≈ -6.6%，Tc ≈ 0% —— 全部绿色
    """
    rest_sigma = 2.5
    base = {'LMM': 26.0, 'RMM': 25.3, 'LTA': 29.0, 'RTA': 29.6}
    # 三次咬合整体强度略有自然变化（不改变通道间比例，指标不受影响）
    seg_scale = [0.93, 1.0, 0.97]

    channels = {}
    for name, b0 in base.items():
        targets = [b0 * s * rng.uniform(0.97, 1.03) for s in seg_scale]
        channels[name] = build_channel(
            rng, name, targets,
            rest_sigma=rest_sigma,
            fatigue=(1.0, 0.88),       # 第三次咬合轻微减弱（自然疲劳，比例不变）
        )

    data = np.column_stack([
        channels['LMM'],
        build_baseline(rng, N_ROWS, 0.5, 0.5),
        build_baseline(rng, N_ROWS, 2.0, 0.5),
        channels['LTA'],
        channels['RMM'],
        channels['RTA'],
    ])
    return data


# ============ TMD 咬合模式 ============
def generate_tmd(rng):
    """
    颞下颌关节紊乱（TMD）模式：右侧为患侧。
    目标 RMS：患侧(右)约为健侧 1.6~1.8 倍 → 不对称指数显著超标（异常对照）。
    指标预期：As-MM ≈ -28%（红）、As-TA ≈ -22%（红）、Tc ≈ -25%（红）、Ac ≈ -3%
    异常特征：患侧静息张力高、早启动 100~200ms、节律不规则、震颤、后段疲劳+漂移+间断静默
    """
    lag = int(rng.uniform(50, 100))    # 患侧早启动 100~200ms
    rest_sigma_healthy = 2.5
    rest_sigma_affected = 3.0          # 患侧静息张力增高（静息患/健比约 1.3~1.6）

    base = {'LMM': 20.5, 'RMM': 36.5, 'LTA': 23.0, 'RTA': 38.0}
    affected = {'RMM', 'RTA'}

    channels = {}
    for name, b0 in base.items():
        is_aff = name in affected
        targets = [b0 * rng.uniform(0.97, 1.03) for _ in range(3)]
        # 咬合3 疲劳：双侧幅值下降，患侧更重（患侧降 20%，健侧降 8%）
        targets[2] = b0 * (0.80 if is_aff else 0.92)

        ch = build_channel(
            rng, name, targets,
            rest_sigma=rest_sigma_affected if is_aff else rest_sigma_healthy,
            lag=lag if is_aff else 0,          # 患侧早启动
            fatigue=(1.0, 0.55) if is_aff else (1.0, 0.85),
            drift_amp=(6.0 if name == 'RMM' else 8.0) if is_aff else 3.0,
            irregular=is_aff,                   # 患侧节律不规则
            tremor_amp=0.8 if is_aff else 0.0,  # 患侧震颤
            onset_immediate=is_aff,             # 患侧确定性早启动
        )

        # 患侧咬合3 加入间断肌电静默（校准前置零，校准后保持目标 RMS）
        if is_aff:
            a, b = BITE_INTERVALS[2]
            sil = np.ones(b - a)
            pos = 0
            while pos < len(sil):
                sl = int(rng.uniform(80, 250))
                gp = int(rng.uniform(400, 1200))
                sl = min(sl, len(sil) - pos)
                if sl > 0:
                    sil[pos:pos + sl] = 0.0
                pos += sl + gp
            seg_view = ch[a:b]
            # 静默作用后重新校准（保持疲劳段目标 RMS）
            ch[a:b] = calibrate_rms(seg_view * sil, targets[2])
        channels[name] = ch

    data = np.column_stack([
        channels['LMM'],
        build_baseline(rng, N_ROWS, 0.5, 0.5),
        build_baseline(rng, N_ROWS, 2.0, 0.6),
        channels['LTA'],
        channels['RMM'],
        channels['RTA'],
    ])
    return data


def build_baseline(rng, n, base, tol):
    """基线通道：直流偏置 + 小随机抖动（量级与模板一致）。"""
    return base + rng.uniform(-tol, tol, n)


# ============ 扰动注入（鲁棒性扫描：工程类 + 采集类扰动） ============
EMG_COLS = ['LMM', 'LTA', 'RMM', 'RTA']   # 4 条肌电通道（基线列不参与扰动）


def apply_noise(data, sigma, rng, seed=0):
    """单侧加性噪声：sigma 叠加在右侧肌电通道（模拟右侧电极接触不良/噪声增大）。"""
    d = data.copy()
    for ch in ['RMM', 'RTA']:
        d[:, COL[ch]] = d[:, COL[ch]] + rng.normal(0, sigma, N_ROWS)
    return d


def apply_clipping(data, limit, rng, seed=0):
    """单侧削顶/饱和：右侧肌电通道幅值裁剪到 [-limit, +limit]（模拟单侧放大器饱和）。"""
    d = data.copy()
    for ch in ['RMM', 'RTA']:
        d[:, COL[ch]] = np.clip(d[:, COL[ch]], -limit, limit)
    return d


def apply_crosstalk(data, ratio, rng, seed=0):
    """单侧电极串扰：右侧咬肌/颞肌互相泄漏 ratio 能量（模拟患侧电极位移/耦合）。"""
    d = data.copy()
    c1, c2 = 'RMM', 'RTA'
    d[:, COL[c1]] = d[:, COL[c1]] + ratio * d[:, COL[c2]]
    d[:, COL[c2]] = d[:, COL[c2]] + ratio * d[:, COL[c1]]
    return d


def apply_gain_mismatch(data, ratio, rng, seed=0):
    """通道增益不一致：右侧整体增益放大 (1+ratio)，左侧保持 1.0。"""
    d = data.copy()
    for ch in ['RMM', 'RTA']:
        d[:, COL[ch]] = d[:, COL[ch]] * (1.0 + ratio)
    return d


# 扰动注册表：name -> (函数, 扫描强度列表, 横轴单位)
DISTURBANCES = {
    '加性噪声(σ)':       (apply_noise,         [0, 2, 4, 6, 8, 12, 16, 22, 30, 40], '噪声σ'),
    '削顶饱和(limit)':   (apply_clipping,      [160, 120, 95, 78, 64, 52, 42, 34, 27, 21], '幅值上限'),
    '电极串扰(ratio)':   (apply_crosstalk,     [0, 0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.52, 0.65, 0.80], '串扰比'),
    '增益不一致(ratio)': (apply_gain_mismatch, [0, 0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.52, 0.65, 0.80], '增益差'),
}


def _find_failure_threshold(x_vals, y_arr, limit):
    """找首个 |y| 越过 limit 的扰动强度（失效阈值）。返回索引或 None。"""
    for i, v in enumerate(y_arr):
        if abs(v) >= limit:
            return i
    return None


def run_robustness_scan(mode='normal', dist_name='加性噪声(σ)', seed=42):
    """
    鲁棒性扫描：固定 seed 生成基准数据，按扰动强度扫描，计算各强度下指标。
    返回 (dist_name, x_vals, y_matrix, x_unit)：
      y_matrix[i][j] = 第 i 个强度下、第 j 个指标的【平均绝对值】
    """
    fn, x_vals, unit = DISTURBANCES[dist_name]
    set_seed(seed)
    base = generate_normal(_rng) if mode == 'normal' else generate_tmd(_rng)

    y_matrix = []   # 每个强度一行，每列一个指标的平均 |值|
    for strength in x_vals:
        d = fn(base, strength, _rng)
        idx = np.array(compute_indices(d))
        # 取每个指标在 3 个咬合区间的平均绝对值（鲁棒性看偏离量级）
        y_matrix.append(np.abs(idx).mean(axis=0))
    return dist_name, x_vals, np.array(y_matrix), unit


# ============ 多种子统计报告（Results 真实数据） ============
def run_multi_seed_report(n_seeds=100, seed_start=1, out_dir=None):
    """
    跑 n_seeds 个随机种子，产出 Results 章节需要的真实统计数据：
      1. 指标误差汇总：每个模式下 4 项指标在多种子下的 均值±标准差、最小、最大、
         以及「越出参考范围」的比例（漏报/误报率）
      2. 退化曲线数值：4 类扰动、每个强度点、多种子取平均后的 4 项指标值
    全部写入 CSV（UTF-8-SIG，Excel 直接打开），并返回文件路径列表。
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(
            sys.executable if getattr(sys, 'frozen', False) else __file__))
    os.makedirs(out_dir, exist_ok=True)

    names = [nm for (nm, _f, _r, _l) in _METRICS]
    limits = [l for (_n, _f, _r, l) in _METRICS]
    seeds = list(range(seed_start, seed_start + n_seeds))

    produced = []

    # ---------- 1. 指标误差汇总（两种模式） ----------
    # 每个 seed 取「3 个咬合区间指标的平均值」作为该次运行的 4 项指标
    summary_rows = []
    for mode in ('normal', 'tmd'):
        mode_cn = '正常咬合' if mode == 'normal' else 'TMD紊乱咬合'
        all_means = {nm: [] for nm in names}   # 每个指标收集 n_seeds 个值
        for s in seeds:
            set_seed(s)
            data = generate_normal(_rng) if mode == 'normal' else generate_tmd(_rng)
            idx = np.array(compute_indices(data))
            m = idx.mean(axis=0)               # 3 区间平均
            for j, nm in enumerate(names):
                all_means[nm].append(float(m[j]))
        for j, nm in enumerate(names):
            arr = np.array(all_means[nm])
            mean = arr.mean()
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            # 判定：以「平均越界」为异常（与 format_indices 口径一致）
            n_over = int((np.abs(arr) >= limits[j]).sum())
            rate = n_over / len(arr) * 100
            summary_rows.append({
                '模式': mode_cn, '指标': nm, '参考范围': f'±{limits[j]:.0f}%',
                '均值(%)': round(mean, 3), '标准差(%)': round(std, 3),
                '最小(%)': round(float(arr.min()), 3), '最大(%)': round(float(arr.max()), 3),
                f'越界样本数(/{n_seeds})': n_over, '越界率(%)': round(rate, 2),
            })
    summary_path = os.path.join(out_dir, f"指标误差汇总_{n_seeds}种子.csv")
    _write_csv(summary_path, ['模式', '指标', '参考范围', '均值(%)', '标准差(%)',
                              '最小(%)', '最大(%)', f'越界样本数(/{n_seeds})', '越界率(%)'],
               summary_rows)
    produced.append(summary_path)

    # ---------- 2. 退化曲线数值（4 类扰动 × 多种子平均） ----------
    for dist_name in DISTURBANCES:
        fn, x_vals, unit = DISTURBANCES[dist_name]
        # 每种扰动、每个强度、每个指标：收集 n_seeds 个「平均绝对值」
        # 用固定基准 seed=1 生成一次基准数据，避免每个 seed 重复生成基准（省时）
        set_seed(1)
        base = generate_normal(_rng)
        # 对每个 seed，用同一份 base 数据 + 该 seed 的随机扰动
        rows = []
        for strength in x_vals:
            acc = np.zeros(len(names))
            for s in seeds:
                set_seed(s)
                d = fn(base, strength, _rng)
                idx = np.array(compute_indices(d))
                acc += np.abs(idx).mean(axis=0)
            acc /= len(seeds)
            row = {'扰动强度': strength}
            for j, nm in enumerate(names):
                row[nm] = round(float(acc[j]), 3)
            rows.append(row)
        csv_path = os.path.join(out_dir, f"退化曲线_{dist_name}_{n_seeds}种子均值.csv")
        _write_csv(csv_path, ['扰动强度'] + names, rows)
        produced.append(csv_path)

    return produced


def _write_csv(path, headers, rows):
    """写 CSV（UTF-8-SIG，兼容 Excel）。rows 为 list[dict] 或 list[list]。"""
    with open(path, 'w', encoding='utf-8-sig') as f:
        f.write(",".join(str(h) for h in headers) + "\n")
        for r in rows:
            if isinstance(r, dict):
                f.write(",".join(str(r.get(h, '')) for h in headers) + "\n")
            else:
                f.write(",".join(str(v) for v in r) + "\n")
    return path


# ============ 写文件 ============
HEADER = "LMM,Channel 2,Channel 3,LTA,RMM,RTA"


def write_txt(data, out_path):
    """写出 txt，严格按模板格式：表头 + 6 列，逗号分隔。"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(HEADER + "\n")
        for row in data:
            f.write(",".join(f"{v:.1f}" for v in row) + "\n")


def plot_degradation_svg(dist_name, x_vals, y_matrix, unit, mode, out_path):
    """退化曲线 SVG（零第三方依赖；中文完美渲染，可直接插入 Word / 浏览器打开）。"""
    names = [nm for (nm, _f, _r, _l) in _METRICS]
    limits = [l for (_n, _f, _r, l) in _METRICS]

    W, H = 900, 560
    pad_l, pad_r, pad_t, pad_b = 80, 40, 64, 70
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b

    xmin, xmax = float(min(x_vals)), float(max(x_vals))
    if xmax == xmin:
        xmax = xmin + 1
    ymax = max(25.0, float(np.max(y_matrix)) * 1.15)
    ymax = min(ymax, 100.0)

    def px(x):
        return pad_l + (x - xmin) / (xmax - xmin) * plot_w

    def py(y):
        return pad_t + plot_h - (y / ymax) * plot_h

    colors = ['#c62828', '#ef6c00', '#1565c0', '#6a1b9a']
    mode_cn = "正常咬合" if mode == 'normal' else "TMD 紊乱咬合"
    font = 'font-family="Microsoft YaHei, PingFang SC, sans-serif"'

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append(f'<rect width="{W}" height="{H}" fill="white"/>')

    # 水平图例（标题下方一行）
    lx = 40
    for j, (nm, lim) in enumerate(zip(names, limits)):
        c = colors[j % len(colors)]
        yy = 46
        xx = lx + j * 212
        parts.append(f'<line x1="{xx}" y1="{yy}" x2="{xx+26}" y2="{yy}" stroke="{c}" stroke-width="2.5"/>')
        parts.append(f'<text {font} x="{xx+32}" y="{yy+4}" font-size="12" fill="#333">{nm}（参考 ±{lim:.0f}%）</text>')

    # 网格 + y 刻度
    gy = 0
    while gy <= ymax:
        yp = py(gy)
        parts.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{pad_l+plot_w}" y2="{yp:.1f}" stroke="#e8e8e8" stroke-width="1"/>')
        parts.append(f'<text {font} x="{pad_l-10}" y="{yp+4:.1f}" text-anchor="end" font-size="12" fill="#555">{gy:g}</text>')
        gy += 10

    # x 刻度
    for gx in x_vals:
        parts.append(f'<text {font} x="{px(gx):.1f}" y="{pad_t+plot_h+24}" text-anchor="middle" font-size="12" fill="#555">{gx:g}</text>')

    # 参考线（各指标上限，虚线）
    for j, lim in enumerate(limits):
        yp = py(lim)
        parts.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{pad_l+plot_w}" y2="{yp:.1f}" '
                     f'stroke="{colors[j % len(colors)]}" stroke-width="1" stroke-dasharray="6 4" opacity="0.55"/>')

    # 坐标轴
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#333" stroke-width="1.2"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#333" stroke-width="1.2"/>')

    # 退化曲线 + 数据点 + 失效阈值 × 标记
    for j in range(y_matrix.shape[1]):
        c = colors[j % len(colors)]
        pts = " ".join(f"{px(x):.1f},{py(y_matrix[i, j]):.1f}" for i, x in enumerate(x_vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.2"/>')
        for i, x in enumerate(x_vals):
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y_matrix[i, j]):.1f}" r="3.5" fill="{c}"/>')
        fi = _find_failure_threshold(x_vals, y_matrix[:, j], limits[j])
        if fi is not None:
            fx, fy = px(x_vals[fi]), py(y_matrix[fi, j])
            parts.append(f'<line x1="{fx-7:.1f}" y1="{fy-7:.1f}" x2="{fx+7:.1f}" y2="{fy+7:.1f}" stroke="{c}" stroke-width="2.5"/>')
            parts.append(f'<line x1="{fx-7:.1f}" y1="{fy+7:.1f}" x2="{fx+7:.1f}" y2="{fy-7:.1f}" stroke="{c}" stroke-width="2.5"/>')

    # 标题与轴标签
    parts.append(f'<text {font} x="{W/2}" y="26" text-anchor="middle" font-size="17" font-weight="500" fill="#222">{mode_cn} · {dist_name} — 退化曲线与失效阈值</text>')
    parts.append(f'<text {font} x="{pad_l+plot_w/2}" y="{H-22}" text-anchor="middle" font-size="13" fill="#333">扰动强度（{unit}）→</text>')
    parts.append(f'<text {font} x="26" y="{pad_t+plot_h/2}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 26 {pad_t+plot_h/2})">指标平均绝对值（%）</text>')

    parts.append('</svg>')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(parts))
    return out_path


def scan_all_disturbances(mode='normal', seed=42, out_dir=None):
    """
    对全部 4 类扰动依次扫描，输出退化曲线 PNG + 失效阈值 CSV。
    返回 [(dist_name, png_path, csv_path, threshold_report), ...]
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(
            sys.executable if getattr(sys, 'frozen', False) else __file__))
    os.makedirs(out_dir, exist_ok=True)

    names = [nm for (nm, _f, _r, _l) in _METRICS]
    mode_cn = "正常咬合" if mode == 'normal' else "TMD紊乱咬合"
    results = []
    for dist_name in DISTURBANCES:
        _dn, x_vals, y_matrix, unit = run_robustness_scan(mode, dist_name, seed)

        # 失效阈值报告
        limits = [l for (_n, _f, _r, l) in _METRICS]
        report_lines = []
        for j, (nm, lim) in enumerate(zip(names, limits)):
            fi = _find_failure_threshold(x_vals, y_matrix[:, j], lim)
            if fi is None:
                report_lines.append(f"    {nm}（±{lim:.0f}%）：全程未失效")
            else:
                report_lines.append(f"    {nm}（±{lim:.0f}%）：在 {dist_name}={x_vals[fi]} 处失效")
        report = "\n".join(report_lines)

        # 退化曲线 SVG（零依赖，中文完美渲染）
        svg_path = os.path.join(out_dir, f"退化曲线_{mode_cn}_{dist_name}.svg")
        plot_degradation_svg(dist_name, x_vals, y_matrix, unit, mode, svg_path)

        # 失效阈值 CSV
        csv_path = os.path.join(out_dir, f"失效阈值_{mode_cn}_{dist_name}.csv")
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write(f"扰动强度({unit})," + ",".join(names) + "\n")
            for i, x in enumerate(x_vals):
                f.write(f"{x}," + ",".join(f"{v:.2f}" for v in y_matrix[i]) + "\n")

        results.append((dist_name, svg_path, csv_path, report))
    return results


def generate(mode, seed=None, out_dir=None):
    """生成指定模式的数据并写入 txt。返回 (输出文件路径, 指标, 生理特征文本)。"""
    set_seed(seed)

    if mode == 'normal':
        data = generate_normal(rng=_rng)
        prefix = "正常咬合"
    elif mode == 'tmd':
        data = generate_tmd(rng=_rng)
        prefix = "TMD紊乱咬合"
    else:
        raise ValueError(f"未知模式: {mode}")

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False)
                                                 else __file__))
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{prefix}_{ts}.txt")
    write_txt(data, out_path)

    indices = compute_indices(data)
    physio = compute_physio_features(data)
    return out_path, indices, physio


# ============ GUI ============
class App:
    def __init__(self, root):
        self.root = root
        root.title("肌电模拟数据生成器")
        root.geometry("560x460")
        root.resizable(False, False)

        pad = 20
        frame = tk.Frame(root, padx=pad, pady=pad)
        frame.pack(fill="both", expand=True)

        title = tk.Label(frame, text="肌电信号模拟数据生成器",
                         font=("Microsoft YaHei", 16, "bold"))
        title.pack(pady=(0, 6))

        sub = tk.Label(frame,
                       text="6列 × 10000行 · 500Hz · 输出 .txt（供 EMG-Analyzer 分析）",
                       font=("Microsoft YaHei", 9), fg="#666666")
        sub.pack(pady=(0, 14))

        # 输出目录选择
        dir_frame = tk.Frame(frame)
        dir_frame.pack(fill="x", pady=(0, 10))
        tk.Label(dir_frame, text="输出目录：", font=("Microsoft YaHei", 10)).pack(side="left")
        self.out_dir = tk.StringVar(value=os.path.dirname(os.path.abspath(
            sys.executable if getattr(sys, 'frozen', False) else __file__)))
        self.dir_entry = tk.Entry(dir_frame, textvariable=self.out_dir, width=40)
        self.dir_entry.pack(side="left", padx=6, fill="x", expand=True)
        tk.Button(dir_frame, text="浏览…", command=self.choose_dir).pack(side="left")

        # 随机种子
        seed_frame = tk.Frame(frame)
        seed_frame.pack(fill="x", pady=(0, 16))
        tk.Label(seed_frame, text="随机种子：", font=("Microsoft YaHei", 10)).pack(side="left")
        self.seed_var = tk.StringVar(value="（留空=随机）")
        tk.Entry(seed_frame, textvariable=self.seed_var, width=16).pack(side="left", padx=6)

        # 两个按钮
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=8)
        self.btn_normal = tk.Button(
            btn_frame, text="生成正常咬合数据", width=20, height=2,
            font=("Microsoft YaHei", 11, "bold"),
            bg="#2e7d32", fg="white", activebackground="#1b5e20",
            command=lambda: self.run("normal"))
        self.btn_normal.pack(side="left", padx=10)

        self.btn_tmd = tk.Button(
            btn_frame, text="生成 TMD 紊乱咬合数据", width=20, height=2,
            font=("Microsoft YaHei", 11, "bold"),
            bg="#c62828", fg="white", activebackground="#8e0000",
            command=lambda: self.run("tmd"))
        self.btn_tmd.pack(side="left", padx=10)

        # 鲁棒性扫描按钮
        scan_frame = tk.Frame(frame)
        scan_frame.pack(pady=(6, 0))
        self.btn_scan = tk.Button(
            scan_frame, text="鲁棒性扫描（退化曲线 + 失效阈值）", width=44, height=2,
            font=("Microsoft YaHei", 10, "bold"),
            bg="#1565c0", fg="white", activebackground="#0d47a1",
            command=self.run_scan)
        self.btn_scan.pack()

        # 状态栏
        self.status = tk.Label(frame, text="就绪", fg="#999999",
                               font=("Microsoft YaHei", 9))
        self.status.pack(pady=(14, 0))

    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get())
        if d:
            self.out_dir.set(d)

    def run(self, mode):
        name = "正常咬合" if mode == "normal" else "TMD 紊乱咬合"
        self.status.config(text=f"正在生成{name}数据…", fg="#1565c0")
        self.root.update()

        try:
            seed = self.seed_var.get().strip()
            seed = int(seed) if seed.isdigit() else None
            path, indices, physio = generate(mode, seed=seed, out_dir=self.out_dir.get())
            self.status.config(text=f"完成：{os.path.basename(path)}", fg="#2e7d32")
            msg = f"{name}数据已生成：\n{path}\n\n{format_indices(indices)}\n\n—— 生理特征核对 ——\n{physio}"
            if mode == 'normal':
                messagebox.showinfo("生成完成", msg)
            else:
                messagebox.showwarning("生成完成（TMD 异常对照）", msg)
        except Exception as e:
            self.status.config(text="生成失败", fg="#c62828")
            messagebox.showerror("错误", f"生成失败：\n{e}")

    def run_scan(self):
        """鲁棒性扫描：4 类扰动 × 10 强度 → 退化曲线 SVG + 失效阈值 CSV。"""
        self.status.config(text="鲁棒性扫描中（4 类扰动 × 10 个强度）…", fg="#1565c0")
        self.root.update()

        try:
            seed = self.seed_var.get().strip()
            seed = int(seed) if seed.isdigit() else 42
            out_dir = self.out_dir.get()
            results = scan_all_disturbances('normal', seed=seed, out_dir=out_dir)
            self.status.config(text="扫描完成", fg="#2e7d32")

            lines = ["鲁棒性扫描完成（正常咬合基准）：", ""]
            for dist_name, svg, csv, report in results:
                lines.append(f"【{dist_name}】")
                lines.append(report)
                lines.append("")
            lines.append(f"输出目录：{out_dir}")
            lines.append("每类扰动生成：退化曲线.svg + 失效阈值.csv")
            messagebox.showinfo("鲁棒性扫描完成", "\n".join(lines))
            try:
                os.startfile(out_dir)   # 打开输出目录方便查看
            except Exception:
                pass
        except Exception as e:
            self.status.config(text="扫描失败", fg="#c62828")
            messagebox.showerror("错误", f"扫描失败：\n{e}")


def main():
    # 多种子统计报告 CLI（Results 真实数据）：
    # 用法：python EMG_Data_Generator.py report [n_seeds] [seed_start] [out_dir]
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
        seed_start = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1
        out_dir = sys.argv[4] if len(sys.argv) > 4 else None
        paths = run_multi_seed_report(n_seeds=n_seeds, seed_start=seed_start, out_dir=out_dir)
        print(f"已生成 {len(paths)} 个 Results 数据文件（{n_seeds} 种子，起始 {seed_start}）：")
        for p in paths:
            print("  " + p)
        return

    # 鲁棒性扫描 CLI（有无 GUI 环境均可用）：
    # 用法：python EMG_Data_Generator.py scan [seed] [out_dir]
    if len(sys.argv) > 1 and sys.argv[1] == 'scan':
        seed = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 42
        out_dir = sys.argv[3] if len(sys.argv) > 3 else None
        results = scan_all_disturbances('normal', seed=seed, out_dir=out_dir)
        for dist_name, svg, csv, report in results:
            print(f"【{dist_name}】")
            print(report)
            print(f"  退化曲线: {svg}")
            print(f"  失效阈值表: {csv}")
            print()
        return

    if not _HAS_TK:
        # 无 GUI 环境：支持命令行调用
        # 用法：python EMG_Data_Generator.py [normal|tmd] [seed] [out_dir]
        mode = sys.argv[1] if len(sys.argv) > 1 else None
        if mode not in ('normal', 'tmd'):
            print("用法: python EMG_Data_Generator.py [normal|tmd] [seed] [out_dir]")
            print("  normal = 生成正常咬合数据")
            print("  tmd    = 生成 TMD 紊乱咬合数据")
            sys.exit(1)
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
        out_dir = sys.argv[3] if len(sys.argv) > 3 else None
        path, indices, physio = generate(mode, seed=seed, out_dir=out_dir)
        print(f"已生成: {path}")
        print(format_indices(indices))
        print("\n—— 生理特征核对 ——")
        print(physio)
        return

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
