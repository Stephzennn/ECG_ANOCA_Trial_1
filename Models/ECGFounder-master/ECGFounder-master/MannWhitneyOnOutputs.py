import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt

from embedding import loadTaskNames
from LogisticRegressionOnOutputs import base_path, taskPath, biomarkerClasses, getPersonLogits


# Bivariate analysis with the Mann-Whitney U test (Wilcoxon rank sum) instead of logistic regression:
# for each class, are the logits of label = 1 people ranked higher (or lower) than those of label = 0 people?
# Rank based, so no normality or linear log odds assumption and outliers don't pull it around.
# Ranks don't change when the logit is z scored, so the raw logits are used.
#   AUC = U / (n1 * n0), the same AUC as roc_auc_score (the chance a random label = 1 person has the higher logit)
#   rank biserial r = 2 * AUC - 1, from -1 to 1 (0 = no difference)
# p_FDR is the Benjamini-Hochberg corrected p value, over the classes tested in the table (150 or the biomarkerClasses).
def bivariateMannWhitney(table, label = "ANOCA", tasks = None):
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    y = table[label]
    rows = []
    for task in tasks:
        positive = table.loc[y == 1, task]
        negative = table.loc[y == 0, task]
        U, p = mannwhitneyu(positive, negative, alternative = "two-sided")
        auc = U / (len(positive) * len(negative))
        rows.append({
            "class": task,
            "median_pos": positive.median(),
            "median_neg": negative.median(),
            "U": U,
            "AUC": auc,
            "rank_biserial": 2 * auc - 1,
            "p": p,
        })

    results = pd.DataFrame(rows).set_index("class")
    results["p_FDR"] = multipletests(results["p"], method = "fdr_bh")[1]
    return results.sort_values("p")


# Same picture style as showBivariateTable in LogisticRegressionOnOutputs.py.
# Red rows: higher logit in label = 1 (AUC > 0.5), blue rows: lower logit in label = 1.
def showMannWhitneyTable(results, label = "ANOCA", top_n = 20, title = "", outputPath = None):
    top = results.head(top_n)

    show = pd.DataFrame(index = top.index)
    show[f"median {label}+"] = [f"{m:.2f}" for m in top["median_pos"]]
    show[f"median {label}-"] = [f"{m:.2f}" for m in top["median_neg"]]
    show["U"] = [f"{u:.0f}" for u in top["U"]]
    show["AUC"] = [f"{a:.2f}" for a in top["AUC"]]
    show["rank biserial r"] = [f"{r:+.2f}" for r in top["rank_biserial"]]
    show["p"] = [f"{p:.3f}" for p in top["p"]]
    show["FDR p"] = [f"{p:.3f}" for p in top["p_FDR"]]

    fig, ax = plt.subplots(figsize = (14, 0.4 * len(show) + 1))
    ax.axis("off")
    tbl = ax.table(cellText = show.values, rowLabels = show.index, colLabels = show.columns,
                   loc = "center", cellLoc = "center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)

    # color each row by direction of the effect (row -1 is the row label column)
    for i, auc in enumerate(top["AUC"]):
        color = "#fde0dd" if auc > 0.5 else "#deebf7"
        for j in range(-1, len(show.columns)):
            tbl[(i + 1, j)].set_facecolor(color)

    fig.suptitle(title, fontsize = 11, y = 0.98)
    fig.text(0.5, 0.02,
             f"Red: higher logit in {label}+ (AUC > 0.5).  Blue: lower logit in {label}+ (AUC < 0.5).  "
             "Two sided Mann-Whitney U; FDR p = Benjamini-Hochberg over all classes tested in this table.",
             ha = "center", fontsize = 8, style = "italic")
    if outputPath is not None:
        plt.savefig(outputPath, dpi = 200, bbox_inches = "tight")
    plt.show()
    return show


if __name__ == "__main__":

    personTable = getPersonLogits()
    label = "ANOCA"

    missing = [c for c in biomarkerClasses if c not in loadTaskNames(taskPath)]
    assert not missing, f"Not in tasks.txt: {missing}"

    # every phase in one run, since the test is fast (no model fitting)
    for phase in ["rest", "stress", "recovery"]:
        phaseTable = personTable.xs(phase, level = "phase")
        nPos, nNeg = phaseTable[label].sum(), (phaseTable[label] == 0).sum()
        print(f"\n===== {phase}, {label}: {nPos} positive, {nNeg} negative =====")

        # all 150 classes (FDR over 150)
        mwTable = bivariateMannWhitney(phaseTable, label)
        mwPath = base_path / "csv" / f"ScienceDB_mannwhitney_{phase}_{label}.csv"
        mwTable.to_csv(mwPath)
        print(f"Mann-Whitney U, top 20 by p (full table saved to {mwPath}):")
        print(mwTable.head(20).round(3))
        print(f"Classes with p < 0.05: {(mwTable['p'] < 0.05).sum()}, with FDR p < 0.05: {(mwTable['p_FDR'] < 0.05).sum()}")

        showMannWhitneyTable(
            mwTable, label, top_n = 20,
            title = f"{phase.capitalize()} ECG: {label}+ vs {label}- ({nPos} vs {nNeg}), "
                    f"top 20 of {len(mwTable)} ECGFounder class logits, Mann-Whitney U each (FDR over {len(mwTable)})",
            outputPath = base_path / "Images" / f"mannwhitney_{phase}_{label}_table.png")

        # only the pre-specified ECG biomarker classes (FDR over these only)
        biomarkerTable = bivariateMannWhitney(phaseTable, label, tasks = biomarkerClasses)
        biomarkerTable.to_csv(base_path / "csv" / f"ScienceDB_mannwhitney_biomarkers_{phase}_{label}.csv")
        print(f"\nPre-specified ECG biomarkers ({len(biomarkerClasses)} classes):")
        print(biomarkerTable.round(3))
        print(f"Biomarkers with p < 0.05: {(biomarkerTable['p'] < 0.05).sum()}, with FDR p < 0.05: {(biomarkerTable['p_FDR'] < 0.05).sum()}")

        showMannWhitneyTable(
            biomarkerTable, label, top_n = len(biomarkerTable),
            title = f"{phase.capitalize()} ECG: {label}+ vs {label}- ({nPos} vs {nNeg}), "
                    f"{len(biomarkerClasses)} pre-specified ECG biomarker logits, Mann-Whitney U (FDR over these {len(biomarkerClasses)})",
            outputPath = base_path / "Images" / f"mannwhitney_biomarkers_{phase}_{label}_table.png")
