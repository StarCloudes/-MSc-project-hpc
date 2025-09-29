#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 pandas 读取 x,pdf CSV，生成：
1) 左：Portfolio Loss PDF（可选对数 y 轴）
2) 右：Survival Function 1-CDF(x)（对数 y 轴，专看右尾）
并计算/导出 VaR95/99、ES95/99 指标表。
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_pdf_csv(path):
    """
    Load a CSV file and return only the first two numeric columns as x and pdf.
    This version is tolerant to extra columns and bad lines.
    """
    import pandas as pd
    
    try:
        # pandas >= 1.3 可以用 on_bad_lines
        df = pd.read_csv(
            path,
            header=None,           # 不把第一行当成表头
            engine="python",       # 宽松解析
            on_bad_lines="skip"    # 跳过列数不对的行
        )
    except TypeError:
        # pandas < 1.3 用旧参数名
        df = pd.read_csv(
            path,
            header=None,
            engine="python",
            error_bad_lines=False
        )
    
    # 转成数值类型，非数值转 NaN 然后丢掉
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    
    # 只取前两列
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ["x", "pdf"]
    else:
        raise ValueError(f"{path} 没有足够的数值列（至少需要两列）")
    
    return df

def pdf_to_cdf(df: pd.DataFrame) -> pd.Series:
    x = df["x"].to_numpy()
    y = df["pdf"].to_numpy()
    # 梯形累计积分
    cdf = np.zeros_like(y)
    if len(x) > 1:
        cdf[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * np.diff(x))
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
    return pd.Series(cdf, index=df.index, name="cdf")

def var_es_from_pdf(df: pd.DataFrame, alpha: float = 0.95):
    """
    用 CDF 线性插值求 VaR；ES 使用尾部 X*pdf 的积分 / 尾部质量（数值稳健）。
    """
    x = df["x"].to_numpy()
    y = df["pdf"].to_numpy()
    cdf = pdf_to_cdf(df).to_numpy()
    if len(x) == 0:
        return np.nan, np.nan

    # VaR：在 CDF 上插 alpha
    var = np.interp(alpha, cdf, x)

    # 为了精确积分首段：构造从 var 开始的尾部 (xt, yt)
    y_var = float(np.interp(var, x, y))
    mask = x >= var
    xt = np.concatenate([[var], x[mask]])
    yt = np.concatenate([[y_var], y[mask]])
    # 尾部概率质量
    tail_mass = np.trapz(yt, xt)
    if tail_mass <= 0:
        return var, var  # 退化：用 VaR 代替 ES

    # 条件期望：∫ x*y dx / ∫ y dx
    num = np.trapz(xt * yt, xt)
    es = num / tail_mass
    return var, es

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True, help="输出图片路径，如 out/loss_pdfs_tail.png")
    ap.add_argument("--summary", default=None, help="指标汇总 CSV 输出（默认与图片同目录 metrics_summary.csv）")
    ap.add_argument("--logy", action="store_true", help="PDF 图使用对数 y 轴")
    ap.add_argument("--xlim", nargs=2, type=float, help="PDF 子图 x 轴范围，例如 --xlim 1e8 1e9")
    ap.add_argument("--labels", nargs="*", help="曲线标签，数量需与 CSV 数一致")
    ap.add_argument("--alpha95", type=float, default=0.95, help="VaR/ES 的低分位（默认 0.95）")
    ap.add_argument("--alpha99", type=float, default=0.99, help="VaR/ES 的高分位（默认 0.99）")
    ap.add_argument("csvs", nargs="+", help="若干 x,pdf CSV 文件")
    args = ap.parse_args()

    ensure_dir(args.out)
    if args.summary is None:
        base_dir = os.path.dirname(os.path.abspath(args.out))
        args.summary = os.path.join(base_dir, "metrics_summary.csv")
    ensure_dir(args.summary)

    K = len(args.csvs)
    labels = args.labels if (args.labels and len(args.labels) == K) else [os.path.splitext(os.path.basename(p))[0] for p in args.csvs]

    series = [load_pdf_csv(p) for p in args.csvs]

    # 计算指标
    rows = []
    for lab, df in zip(labels, series):
        v95, es95 = var_es_from_pdf(df, args.alpha95)
        v99, es99 = var_es_from_pdf(df, args.alpha99)
        rows.append({"label": lab, "VaR95": v95, "ES95": es95, "VaR99": v99, "ES99": es99})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary, index=False)
    print("Saved metrics:", args.summary)
    print(summary)

    # 画图：左 PDF；右 1-CDF（log y）
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # PDF
    for lab, df, met in zip(labels, series, rows):
        ax[0].plot(df["x"], df["pdf"], label=lab)
        ax[0].axvline(met["VaR95"], ls="--", alpha=0.35)
        ax[0].axvline(met["VaR99"], ls=":",  alpha=0.35)
    ax[0].set_title("Portfolio Loss PDF")
    ax[0].set_xlabel("Portfolio Loss")
    ax[0].set_ylabel("PDF")
    if args.logy:
        ax[0].set_yscale("log")
    if args.xlim:
        ax[0].set_xlim(args.xlim[0], args.xlim[1])
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Survival function 1-CDF(x)
    for lab, df in zip(labels, series):
        cdf = pdf_to_cdf(df)
        surv = 1.0 - cdf
        ax[1].plot(df["x"], surv, label=lab)
    ax[1].set_title("Survival Function 1 - CDF(x)")
    ax[1].set_xlabel("Portfolio Loss")
    ax[1].set_ylabel("1 - CDF(x)")
    ax[1].set_yscale("log")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved figure: {args.out}")

if __name__ == "__main__":
    main()
