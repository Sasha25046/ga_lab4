import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("convergence_long.csv")
for ds, g in df.groupby("Dataset"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for cfg, c in g.groupby("Config"):
        ax[0].plot(c["Evals"], c["AvgBestFitness"], label=cfg, linewidth=2)
        ax[1].plot(c["Evals"], c["AvgFeasibleFraction"], label=cfg, linewidth=2)
    ax[0].set_yscale("symlog", linthresh=10)
    ax[0].set_title(f"Динаміка fitness, {ds} (symlog)")
    ax[1].set_title(f"Частка допустимих особин (hard = 0), {ds}")
    for a in ax:
        a.set_xlabel("Кількість оцінок fitness")
        a.grid(True, linestyle="--", alpha=0.6)
        a.legend()
    plt.tight_layout()
    plt.savefig(f"convergence_{ds}.png", dpi=150)
    plt.close(fig)

p1 = pd.read_csv("part1_convergence.csv")
plt.figure(figsize=(7, 4.5))
plt.plot(p1["Evals"], p1["GA_Median"].clip(lower=1e-6), label="ГА (медіана)", linewidth=2)
d = p1.dropna(subset=["GDmulti_Median"])
plt.plot(d["Evals"], d["GDmulti_Median"], label="GD з перезапусками (медіана)", linewidth=2)
plt.yscale("log")
plt.xlabel("Кількість викликів f")
plt.ylabel("Найкраще f")
plt.title("Задача 1: збіжність за бюджетом")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("part1_convergence.png", dpi=150)
print("[OK] графіки збережено")