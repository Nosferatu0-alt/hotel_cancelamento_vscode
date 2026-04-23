
"""
eda.py — Análise Exploratória de Dados (EDA)
Etapa 1 do Desafio 07: gera e salva todos os gráficos exploratórios.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import COLORS, OUTPUT_DIR


def _save(fig: plt.Figure, fname: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight", dpi=120)
    plt.show()
    plt.close(fig)


#  1. Taxa de cancelamento geral 

def plot_cancelamento_geral(df: pd.DataFrame) -> None:
    """Barra + pizza da distribuição geral de cancelamentos."""
    counts = df["is_canceled"].value_counts()
    colors = [COLORS["green"], COLORS["red"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição de Cancelamentos — Amostra", fontsize=14, fontweight="bold")

    bars = axes[0].bar(["Mantida (0)", "Cancelada (1)"], counts.values,
                       color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Contagem de Reservas")
    axes[0].set_ylabel("Quantidade")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{val:,}\n({val / len(df) * 100:.1f}%)",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, max(counts.values) * 1.2)

    wedge_props = dict(width=0.55, edgecolor="white", linewidth=3)
    axes[1].pie(counts.values, labels=["Mantida", "Cancelada"],
                autopct="%1.1f%%", colors=colors, startangle=90,
                wedgeprops=wedge_props, textprops={"fontsize": 12})
    axes[1].set_title("Proporção de Cancelamentos")

    plt.tight_layout()
    _save(fig, "01_taxa_cancelamento.png")
    print(f"Taxa de cancelamento: {counts[1] / len(df) * 100:.2f}%")


#  2. Cancelamento por tipo de hotel 

def plot_cancelamento_hotel(df: pd.DataFrame) -> None:
    """Volume, taxa e stacked bar por tipo de hotel."""
    hotel_cancel = (
        df.groupby("hotel")["is_canceled"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )
    hotel_cancel.columns = ["hotel", "cancelados", "total", "taxa"]
    hotel_cancel["taxa_pct"] = hotel_cancel["taxa"] * 100
    hotel_cancel["mantidos"] = hotel_cancel["total"] - hotel_cancel["cancelados"]

    hotel_colors_map = {"City Hotel": COLORS["blue"], "Resort Hotel": COLORS["orange"]}
    hcolors = [hotel_colors_map.get(h, "#95a5a6") for h in hotel_cancel["hotel"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cancelamentos por Tipo de Hotel", fontsize=14, fontweight="bold")

    axes[0].bar(hotel_cancel["hotel"], hotel_cancel["total"], color=hcolors, edgecolor="white")
    axes[0].set_title("Volume Total de Reservas")
    axes[0].set_ylabel("Quantidade")
    for i, (_, row) in enumerate(hotel_cancel.iterrows()):
        axes[0].text(i, row["total"] + 5, f"{int(row['total']):,}", ha="center", fontsize=11, fontweight="bold")

    bars = axes[1].bar(hotel_cancel["hotel"], hotel_cancel["taxa_pct"], color=hcolors, edgecolor="white")
    axes[1].set_title("Taxa de Cancelamento (%)")
    axes[1].set_ylabel("Cancelamentos (%)")
    axes[1].set_ylim(0, 100)
    axes[1].axhline(hotel_cancel["taxa_pct"].mean(), color="red", linestyle="--", alpha=0.6, label="Média")
    axes[1].legend()
    for bar, val in zip(bars, hotel_cancel["taxa_pct"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

    axes[2].bar(hotel_cancel["hotel"], hotel_cancel["mantidos"], label="Mantidas", color=COLORS["green"], edgecolor="white")
    axes[2].bar(hotel_cancel["hotel"], hotel_cancel["cancelados"], bottom=hotel_cancel["mantidos"],
                label="Canceladas", color=COLORS["red"], edgecolor="white")
    axes[2].set_title("Mantidas vs Canceladas")
    axes[2].set_ylabel("Quantidade")
    axes[2].legend()

    plt.tight_layout()
    _save(fig, "02_cancelamento_hotel.png")
    print(hotel_cancel[["hotel", "total", "cancelados", "taxa_pct"]].to_string(index=False))


#  3. Lead Time

def plot_lead_time(df: pd.DataFrame) -> None:
    """Histograma KDE, boxplot e cancelamento por faixa de lead time."""
    df = df.copy()
    df["lead_time_faixa"] = pd.cut(
        df["lead_time"],
        bins=[0, 7, 30, 90, 180, 365, 999],
        labels=["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", ">365d"],
    )
    cancel_by_faixa = df.groupby("lead_time_faixa")["is_canceled"].mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Análise de Lead Time", fontsize=14, fontweight="bold")

    for status, label, color in [(0, "Mantida", COLORS["green"]), (1, "Cancelada", COLORS["red"])]:
        subset = df[df["is_canceled"] == status]["lead_time"]
        axes[0].hist(subset, bins=40, alpha=0.55, color=color, label=label, density=True)
        subset.plot.kde(ax=axes[0], color=color, linewidth=2)
    axes[0].set_title("Distribuição do Lead Time")
    axes[0].set_xlabel("Lead Time (dias)")
    axes[0].set_ylabel("Densidade")
    axes[0].legend()

    groups = [df[df["is_canceled"] == s]["lead_time"].values for s in [0, 1]]
    bp = axes[1].boxplot(groups, labels=["Mantida", "Cancelada"], patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], [COLORS["green"], COLORS["red"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Lead Time por Status")
    axes[1].set_ylabel("Lead Time (dias)")

    cancel_by_faixa.plot(kind="bar", ax=axes[2], color=COLORS["purple"], edgecolor="white")
    axes[2].set_title("Cancelamento por Faixa de Lead Time")
    axes[2].set_xlabel("Faixa")
    axes[2].set_ylabel("Taxa (%)")
    axes[2].tick_params(axis="x", rotation=30)
    for bar in axes[2].patches:
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "03_lead_time.png")
    print(df.groupby("is_canceled")["lead_time"].describe().round(2))


#  4. Correlações

def plot_correlacoes(df: pd.DataFrame) -> None:
    """Top 15 correlações com target + heatmap."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    corr_target = df[num_cols].corr()["is_canceled"].drop("is_canceled").sort_values(key=abs, ascending=False)
    top15 = corr_target.head(15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Correlação com Cancelamento", fontsize=14, fontweight="bold")

    colors_corr = [COLORS["red"] if v > 0 else COLORS["green"] for v in top15.values]
    bars = axes[0].barh(top15.index, top15.values, color=colors_corr, edgecolor="white")
    axes[0].set_title("Top 15 Correlações com is_canceled")
    axes[0].set_xlabel("Correlação de Pearson")
    axes[0].axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, top15.values):
        axes[0].text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    axes[0].invert_yaxis()

    top10_feats = list(corr_target.head(10).index) + ["is_canceled"]
    heatmap_data = df[top10_feats].corr()
    mask = np.triu(np.ones_like(heatmap_data, dtype=bool))
    sns.heatmap(heatmap_data, mask=mask, ax=axes[1], cmap="RdYlGn",
                annot=True, fmt=".2f", square=True, linewidths=0.5)
    axes[1].set_title("Mapa de Calor — Top 10 Features")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save(fig, "04_correlacoes.png")


#  5. Variáveis categóricas 

def plot_categoricas(df: pd.DataFrame) -> None:
    """Taxa de cancelamento por deposit_type, customer_type, distribution_channel e meal."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Taxa de Cancelamento por Variáveis Categóricas", fontsize=14, fontweight="bold")

    pairs = [
        ("deposit_type",         "Tipo de Depósito",          axes[0, 0], COLORS["red"]),
        ("customer_type",        "Tipo de Cliente",            axes[0, 1], COLORS["blue"]),
        ("distribution_channel", "Canal de Distribuição",      axes[1, 0], COLORS["purple"]),
        ("meal",                 "Tipo de Refeição",           axes[1, 1], COLORS["orange"]),
    ]

    for col, title, ax, color in pairs:
        data = df.groupby(col)["is_canceled"].mean() * 100
        if col == "deposit_type":
            bar_colors = [COLORS["red"] if v > 50 else COLORS["green"] for v in data.values]
        else:
            bar_colors = color
        ax.bar(data.index, data.values, color=bar_colors, edgecolor="white", linewidth=1.5)
        ax.set_title(f"Taxa de Cancelamento — {title}")
        ax.set_ylabel("Cancelamento (%)")
        ax.set_ylim(0, 110)
        if col not in ("deposit_type",):
            ax.axhline(df["is_canceled"].mean() * 100, color="red", linestyle="--", alpha=0.7, label="Média")
            ax.legend()
        for i, val in enumerate(data.values):
            ax.text(i, val + 1.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    _save(fig, "05_categoricas.png")


#  6. Missing e Outliers 

def plot_missing_outliers(df: pd.DataFrame) -> None:
    """Percentual de nulos e detecção de outliers em lead_time."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Nulos": missing, "Percentual (%)": missing_pct}).sort_values("Nulos", ascending=False)
    missing_df = missing_df[missing_df["Nulos"] > 0]

    Q1, Q3 = df["lead_time"].quantile(0.25), df["lead_time"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df["lead_time"] < Q1 - 1.5 * IQR) | (df["lead_time"] > Q3 + 1.5 * IQR)).sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Valores Ausentes e Outliers", fontsize=14, fontweight="bold")

    if len(missing_df) > 0:
        missing_df["Percentual (%)"].plot(kind="barh", ax=axes[0], color=COLORS["red"], edgecolor="white")
        axes[0].set_title("Percentual de Nulos por Coluna")
        axes[0].set_xlabel("Percentual (%)")
    else:
        axes[0].text(0.5, 0.5, "Nenhum valor nulo\nencontrado! ✅",
                     ha="center", va="center", fontsize=14, transform=axes[0].transAxes)
        axes[0].axis("off")

    bp = axes[1].boxplot(df["lead_time"].dropna(), patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2.5})
    bp["boxes"][0].set_facecolor(COLORS["blue"])
    bp["boxes"][0].set_alpha(0.7)
    axes[1].set_title(f"Outliers em Lead Time ({outliers} detectados via IQR)")
    axes[1].set_ylabel("Lead Time (dias)")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["Lead Time"])

    plt.tight_layout()
    _save(fig, "06_missing_outliers.png")
    print(f"\nOutliers em lead_time: {outliers} ({outliers / len(df) * 100:.2f}%)")


# 7. Sazonalidade

def plot_sazonalidade(df: pd.DataFrame) -> None:
    """Taxa de cancelamento por mês e estação."""
    month_cancel = df.groupby("arrival_month_name")["is_canceled"].mean() * 100
    month_cancel = month_cancel.reindex(range(1, 13))

    estacao_order = ["Verão", "Outono", "Inverno", "Primavera"]
    estacao_cancel = df.groupby("estacao")["is_canceled"].mean() * 100
    estacao_cancel = estacao_cancel.reindex(estacao_order)
    season_colors = [COLORS["orange"], COLORS["red"], COLORS["blue"], COLORS["green"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Sazonalidade e Cancelamentos", fontsize=14, fontweight="bold")

    axes[0].plot(range(1, 13), month_cancel.values, "o-", color=COLORS["purple"], linewidth=2, markersize=7)
    axes[0].fill_between(range(1, 13), month_cancel.values, alpha=0.2, color=COLORS["purple"])
    axes[0].set_title("Taxa de Cancelamento por Mês")
    axes[0].set_xlabel("Mês")
    axes[0].set_ylabel("Cancelamento (%)")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"])
    axes[0].grid(True, alpha=0.4)

    bars = axes[1].bar(estacao_cancel.index, estacao_cancel.values, color=season_colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Taxa de Cancelamento por Estação")
    axes[1].set_ylabel("Cancelamento (%)")
    axes[1].axhline(df["is_canceled"].mean() * 100, color="red", linestyle="--", alpha=0.7, label="Média")
    axes[1].legend()
    for bar, val in zip(bars, estacao_cancel.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    _save(fig, "07_sazonalidade.png")


# Orquestrador

def run_eda(df: pd.DataFrame) -> None:
    """Executa toda a EDA em sequência."""
    print("\n" + "=" * 60)
    print("ETAPA 1 — ANÁLISE EXPLORATÓRIA (EDA)")
    print("=" * 60)

    plot_cancelamento_geral(df)
    plot_cancelamento_hotel(df)
    plot_lead_time(df)
    plot_correlacoes(df)
    plot_categoricas(df)
    plot_missing_outliers(df)
    plot_sazonalidade(df)

    print("\n EDA concluída. Gráficos salvos em outputs/")
