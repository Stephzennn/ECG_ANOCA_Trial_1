import numpy as np
import pandas as pd
import torch
from pathlib import Path
from net1d import Net1D
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from matplotlib import pyplot as plt

from embedding import createDataloader, loadWeightsToModel, generateOutput, averageLogitsPerPerson, loadTaskNames


# Paths relative to this file, so it runs the same on Windows and on the cluster
base_path = Path(__file__).resolve().parent
taskPath = base_path / "tasks.txt"
checkpointPath = base_path / "checkpoint" / "12_lead_ECGFounder.pth"
csv_filepath = base_path / "csv" / "ScienceDB_label_all_phases.csv"
ecg_filepath = str(base_path / "data" / "ScienceDB_Data")

# The per person (and phase) 150 logits are saved here, so the model only runs once
personLogitsPath = base_path / "csv" / "ScienceDB_person_logits.csv"


# Pre-specified ECG biomarker classes (fewer tests -> lighter FDR correction).
# Chosen on clinical grounds, not from the results:
#   - only classes that describe the current ECG (waveform, interval, voltage, axis)
#   - no disease labels (infarct, MI, pacemaker, arrhythmia, ...)
#   - no "compared with a previous ECG" classes (NOW EVIDENT IN, NO LONGER EVIDENT IN, HAS REPLACED ...),
#     since each person has no earlier ECG to compare with
#   - no overall summaries (ABNORMAL ECG, NORMAL ECG, BORDERLINE ECG)
#   - weighted towards T wave / repolarization, our main hypothesis
#   - classes that are near duplicates (Spearman r ~0.7-0.95 between person logits) are merged into one,
#     e.g. ST AND T WAVE ABNORMALITY -> T WAVE ABNORMALITY, ST NOW / MORE DEPRESSED -> REPOLARIZATION ABNORMALITY,
#     LEFTWARD AXIS -> LEFT AXIS DEVIATION, NONSPECIFIC INTRAVENTRICULAR BLOCK -> CONDUCTION DELAY
biomarkerClasses = [
    # T wave / ST / repolarization
    "NONSPECIFIC T WAVE ABNORMALITY",
    "WITH REPOLARIZATION ABNORMALITY",      # also stands for ST depression
    "NONSPECIFIC ST ABNORMALITY",
    "EARLY REPOLARIZATION",
    "PROLONGED QT",
    # QRS width, conduction and voltage
    #"WITH QRS WIDENING",
    "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",
    "RSR' PATTERN IN V1",
    "LOW VOLTAGE QRS",
    "VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY",
    # PR interval and P wave
    "WITH 1ST DEGREE AV BLOCK",
    "WITH SHORT PR",
    "NO P-WAVES FOUND",
    # axis
    "LEFT AXIS DEVIATION",
    "RIGHT AXIS DEVIATION",
]


# Run ECGFounder on every 10 second slice, then average the 150 logits per person and phase.
# If this was already done, just read the saved csv.
def getPersonLogits(rerun = False):
    if personLogitsPath.exists() and rerun == False:
        print(f"Reading saved person logits from {personLogitsPath}")
        return pd.read_csv(personLogitsPath, index_col = ["patient_id", "phase"])

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    # 8 workers load and resample the slices in parallel (num_workers=0 made the GPU wait on loading)
    testset, testloader = createDataloader(taskPath, 512, ecg_filepath, csv_filepath, num_workers = 8, ScienceDB=True)
    model = Net1D(
        in_channels=12,
        base_filters=64, #32 64
        ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
        m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose= False,
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=True)
    model.to(device)
    log = loadWeightsToModel(checkpointPath, model, device)
    all_gt, all_embeddings, df_gt, labels, all_pred_prob , all_logits = generateOutput(testloader, model, device)

    personTable = averageLogitsPerPerson(testset, all_logits, taskPath)
    personTable.to_csv(personLogitsPath)
    print(f"Saved person logits to {personLogitsPath}")
    return personTable


# AUC of each of the 150 logits on its own for the label (no model fitting, just ranking).
# AUC < 0.5 means a lower logit goes with the label, so |AUC - 0.5| is the strength.
def univariateAUC(table, label = "ANOCA"):
    y = table[label]
    aucs = {task: roc_auc_score(y, table[task]) for task in loadTaskNames(taskPath)}
    aucs = pd.Series(aucs, name = "AUC").to_frame()
    aucs["strength"] = (aucs["AUC"] - 0.5).abs()
    return aucs.sort_values("strength", ascending = False)


# Logistic regression on the logits (all 150, or only the ones in tasks). L2 penalty with its strength picked
# inside each training fold, so the test fold never helps choose it. Returns the AUC of every test fold.
def logisticModel():
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs = 10, cv = 5, penalty = "l2", scoring = "roc_auc", max_iter = 5000))


# Same seed -> same folds for every call, so the AUCs of two models can be compared fold by fold.
def cvLogisticAUC(table, label = "ANOCA", tasks = None, n_splits = 5, n_repeats = 10, seed = 0):
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    X = table[tasks].to_numpy()
    y = table[label].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = seed)
    return cross_val_score(logisticModel(), X, y, cv = cv, scoring = "roc_auc", n_jobs = -1)


# Is the cross validated AUC better than chance? Shuffle the labels and refit n_permutations times.
def permutationPValue(table, label = "ANOCA", tasks = None, n_permutations = 200, seed = 0):
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    X = table[tasks].to_numpy()
    y = table[label].to_numpy()
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    score, permScores, pValue = permutation_test_score(
        logisticModel(), X, y, cv = cv, scoring = "roc_auc", n_permutations = n_permutations, n_jobs = -1, random_state = seed)
    return score, permScores, pValue


# Bivariate analysis: one logistic regression per class, label ~ that class's logit (150 separate models).
# The logit is z scored first, so beta and OR are per 1 SD and comparable between classes.
# OR > 1: a higher logit for that class goes with label = 1 (e.g. ANOCA+).
# p_FDR is the Benjamini-Hochberg corrected p value, since we run 150 tests.
def bivariateLogistic(table, label = "ANOCA", tasks = None):
    # tasks: which classes to test (default all 150). The FDR correction is over only these classes.
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    y = table[label]
    rows = []
    for task in tasks:
        z = (table[task] - table[task].mean()) / table[task].std()
        X = sm.add_constant(z.rename("logit_z"))
        try:
            m = sm.Logit(y, X).fit(disp = 0)
        except Exception as e:   # e.g. perfect separation, keep going with the other classes
            print(f"SKIP {task}: {e}")
            continue
        beta = m.params["logit_z"]
        ciLow, ciHigh = m.conf_int().loc["logit_z"]
        rows.append({
            "class": task,
            "beta": beta,
            "SE": m.bse["logit_z"],
            "OR_per_SD": np.exp(beta),
            "OR_CI_low": np.exp(ciLow),
            "OR_CI_high": np.exp(ciHigh),
            "p": m.pvalues["logit_z"],
            "AUC": roc_auc_score(y, table[task]),
        })

    results = pd.DataFrame(rows).set_index("class")
    results["p_FDR"] = multipletests(results["p"], method = "fdr_bh")[1]
    return results.sort_values("p")


# Multivariable analysis: one logistic regression, label ~ all the classes in tasks together (z scored).
# Each p is for that class with the other classes held fixed. m.llr_pvalue is the one global test
# (full model vs intercept only), so no FDR is needed for it.
def multivariableLogistic(table, label = "ANOCA", tasks = None):
    # read biomarkerClasses at call time, so editing the list is picked up without re-running this def
    if tasks is None:
        tasks = biomarkerClasses
    Z =(table[tasks] - table[tasks].mean()) / table[tasks].std()
    return sm.Logit(table[label], sm.add_constant(Z)).fit(disp = 0)


# Hosmer-Lemeshow goodness of fit: split people into groups by predicted probability and compare
# observed vs expected label = 1 counts. Large p = no evidence of bad fit.
def hosmerLemeshow(y, prob, groups = 10):
    d = pd.DataFrame({"y": y, "prob": prob})
    d["group"] = pd.qcut(d["prob"], groups, labels = False, duplicates = "drop")
    g = d.groupby("group")
    observed, expected, n = g["y"].sum(), g["prob"].sum(), g.size()
    hl = (((observed - expected) ** 2) / (expected * (1 - expected / n))).sum()
    return hl, chi2.sf(hl, d["group"].nunique() - 2)


# Show the bivariate table as a picture (same style as the tables in DelinationValidation.py).
# One row per class: beta (SE), OR per SD (95% CI), p, FDR p, AUC.
# Red rows: higher logit goes with label = 1, blue rows: lower logit goes with label = 1.
def showBivariateTable(results, top_n = 20, title = "", outputPath = None):
    top = results.head(top_n)

    show = pd.DataFrame(index = top.index)
    show["beta (SE)"] = [f"{b:.2f} ({se:.2f})" for b, se in zip(top["beta"], top["SE"])]
    show["OR per SD (95% CI)"] = [f"{o:.2f} ({lo:.2f}-{hi:.2f})" for o, lo, hi in zip(top["OR_per_SD"], top["OR_CI_low"], top["OR_CI_high"])]
    show["p"] = [f"{p:.3f}" for p in top["p"]]
    show["FDR p"] = [f"{p:.3f}" for p in top["p_FDR"]]
    show["AUC"] = [f"{a:.2f}" for a in top["AUC"]]

    fig, ax = plt.subplots(figsize = (14, 0.4 * len(show) + 1))
    ax.axis("off")
    tbl = ax.table(cellText = show.values, rowLabels = show.index, colLabels = show.columns,
                   loc = "center", cellLoc = "center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)

    # color each row by direction of the effect (row -1 is the row label column)
    for i, orValue in enumerate(top["OR_per_SD"]):
        color = "#fde0dd" if orValue > 1 else "#deebf7"
        for j in range(-1, len(show.columns)):
            tbl[(i + 1, j)].set_facecolor(color)

    # title centered over the whole figure (ax.set_title was pushed right by the wide class name column)
    fig.suptitle(title, fontsize = 11, y = 0.98)
    fig.text(0.5, 0.02,
             "Red: higher logit goes with label = 1 (OR > 1).  Blue: lower logit goes with label = 1 (OR < 1).  "
             "OR per 1 SD of the logit; FDR p = Benjamini-Hochberg over all classes tested in this table.",
             ha = "center", fontsize = 8, style = "italic")
    if outputPath is not None:
        plt.savefig(outputPath, dpi = 200, bbox_inches = "tight")
    plt.show()
    return show


if __name__ == "__main__":

    personTable = getPersonLogits()

    # quick checks: should be 97 people x 3 phases, 150 class columns + n_ECGs + labels
    print(personTable.shape)
    print(personTable.groupby("phase").size())
    print(personTable[["n_ECGs", "ANOCA", "CMD"]].head())

    # one phase at a time: "rest", "stress" or "recovery" (titles and file names follow this)
    phase = "recovery"
    phaseTable = personTable.xs(phase, level = "phase")

    label = "ANOCA"
    print(f"\n{phase}, {label}:{phaseTable[label].sum()} positive, {(phaseTable[label] == 0).sum()} negative")

    # 1) each logit on its own (150 tests, so expect some AUCs around 0.6-0.65 by chance)
    print("\nTop 15 single logits by |AUC - 0.5|:")
    print(univariateAUC(phaseTable, label).head(15).round(3))

    # 1b) bivariate table: ANOCA ~ logit, one model per class, with beta, OR (95% CI) and p values
    bivariateTable = bivariateLogistic(phaseTable, label)
    bivariatePath = base_path / "csv" / f"ScienceDB_bivariate_{phase}_{label}.csv"
    bivariateTable.to_csv(bivariatePath)
    print(f"\nBivariate logistic regression, top 20 by p (full table saved to {bivariatePath}):")
    print(bivariateTable.head(20).round(3))
    print(f"Classes with p < 0.05: {(bivariateTable['p'] < 0.05).sum()}, with FDR p < 0.05: {(bivariateTable['p_FDR'] < 0.05).sum()}")

    showBivariateTable(
        bivariateTable, top_n = 20,
        title = f"{phase.capitalize()} ECG: {label}+ vs {label}- ({phaseTable[label].sum()} vs {(phaseTable[label] == 0).sum()}), "
                f"top 20 of {len(bivariateTable)} ECGFounder class logits, one logistic regression each (FDR over {len(bivariateTable)})",
        outputPath = base_path / "Images" / f"bivariate_{phase}_{label}_table.png")

    # 1c) same bivariate analysis, but only the pre-specified ECG biomarker classes (FDR over these only)
    missing = [c for c in biomarkerClasses if c not in loadTaskNames(taskPath)]
    assert not missing, f"Not in tasks.txt: {missing}"

    biomarkerTable = bivariateLogistic(phaseTable, label, tasks = biomarkerClasses)
    biomarkerTable.to_csv(base_path / "csv" / f"ScienceDB_bivariate_biomarkers_{phase}_{label}.csv")
    print(f"\nPre-specified ECG biomarkers ({len(biomarkerClasses)} classes):")
    print(biomarkerTable.round(3))
    print(f"Biomarkers with p < 0.05: {(biomarkerTable['p'] < 0.05).sum()}, with FDR p < 0.05: {(biomarkerTable['p_FDR'] < 0.05).sum()}")

    showBivariateTable(
        biomarkerTable, top_n = len(biomarkerTable),
        title = f"{phase.capitalize()} ECG: {label}+ vs {label}- ({phaseTable[label].sum()} vs {(phaseTable[label] == 0).sum()}), "
                f"{len(biomarkerClasses)} pre-specified ECG biomarker logits (per 1 SD, FDR over these {len(biomarkerClasses)})",
        outputPath = base_path / "Images" / f"bivariate_biomarkers_{phase}_{label}_table.png")

    # 1d) multivariable logistic regression on the same biomarker classes, all in one model
    m = multivariableLogistic(phaseTable, label)
    print(m.summary())   # coef and p for each class, LLR p-value = global test
    multivariableTable = pd.DataFrame({
        "OR_per_SD": np.exp(m.params),
        "OR_CI_low": np.exp(m.conf_int()[0]),
        "OR_CI_high": np.exp(m.conf_int()[1]),
        "p": m.pvalues}).drop("const")
    multivariableTable.to_csv(base_path / "csv" / f"ScienceDB_multivariable_biomarkers_{phase}_{label}.csv")
    print(multivariableTable.round(3))
    print(f"Global likelihood ratio test (all {len(biomarkerClasses)} vs none): p = {m.llr_pvalue:.3f}")

    hl, hlP = hosmerLemeshow(phaseTable[label], m.predict())
    print(f"Hosmer-Lemeshow goodness of fit: chi2 = {hl:.2f}, p = {hlP:.3f}")

    # 2) logistic regression on all 150 logits, repeated 5 fold cross validation
    foldAUCs = cvLogisticAUC(phaseTable, label)
    print(f"\nLogistic regression, all 150 logits: CV AUC = {foldAUCs.mean():.3f} "
          f"(SD {foldAUCs.std():.3f}, {len(foldAUCs)} folds)")

    # 3) permutation test against chance
    score, permScores, pValue = permutationPValue(phaseTable, label)
    print(f"Permutation test: AUC = {score:.3f}, chance AUC = {permScores.mean():.3f}, p = {pValue:.3f}")

    # 4) same model, but only the pre-specified ECG biomarker logits.
    # If the AUC is close to the 150 logit model, the signal is in these T wave / conduction findings.
    biomarkerAUCs = cvLogisticAUC(phaseTable, label, tasks = biomarkerClasses)
    print(f"\nLogistic regression, {len(biomarkerClasses)} biomarker logits: CV AUC = {biomarkerAUCs.mean():.3f} "
          f"(SD {biomarkerAUCs.std():.3f}, {len(biomarkerAUCs)} folds)")

    score, permScores, pValue = permutationPValue(phaseTable, label, tasks = biomarkerClasses)
    print(f"Permutation test: AUC = {score:.3f}, chance AUC = {permScores.mean():.3f}, p = {pValue:.3f}")

    # fold by fold comparison (both models used the exact same 50 folds)
    difference = biomarkerAUCs - foldAUCs
    print(f"{len(biomarkerClasses)} biomarkers minus all 150: mean AUC difference = {difference.mean():+.3f}, "
          f"biomarker model as good or better in {(difference >= 0).sum()} of {len(difference)} folds")

"""
In recovery ECGs from women with chest pain, a model using 11 pre-specified ECGFounder ECG-biomarker outputs 
discriminated ANOCA+ from ANOCA− with a cross-validated AUC of 0.66 (permutation p = 0.03), matching the full
150-output model. The strongest contributors were nonspecific T-wave abnormality (higher in ANOCA+) and 
nonspecific intraventricular conduction delay (higher in ANOCA−).

"""

"""
Multivariable logistic regression, ANOCA ~ 15 biomarker logits (z scored, OR per 1 SD), one model per phase

                              Rest        Recovery
    n (ANOCA+ / ANOCA-)       97 (63/34)  96 (62/34)
    Global LR test p          0.135       0.026
    Pseudo R2                 0.17        0.22
    Biomarkers with p < 0.05  0           3

Recovery, biomarkers with p < 0.05 (each adjusted for the other 14):

    Biomarker                                       OR per SD (95% CI)   p
    NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY   0.22 (0.07-0.68)     0.008   consistent with bivariate and Mann-Whitney
    NONSPECIFIC ST ABNORMALITY                      2.23 (1.05-4.73)     0.036
    WITH QRS WIDENING                               5.57 (1.01-30.8)     0.049   very wide CI, opposite sign to conduction delay,
                                                                                 likely collinearity, not a finding on its own

Rest Hosmer-Lemeshow: chi2 = 6.67, p = 0.573 (no evidence of poor fit)

In recovery ECGs, the 15 ECGFounder biomarker logits were jointly associated with ANOCA (likelihood ratio
chi2 = 27.4, df = 15, p = 0.026). After mutual adjustment, a higher nonspecific intraventricular conduction delay
logit was associated with lower odds of ANOCA (OR 0.22 per SD, 95% CI 0.07-0.68, p = 0.008), and a higher
nonspecific ST abnormality logit with higher odds (OR 2.23, 1.05-4.73, p = 0.036).

"""

"""
Multivariable logistic regression, ANOCA ~ 14 biomarker logits (z scored, OR per 1 SD), one model per phase
(WITH QRS WIDENING removed: it and the conduction delay logit pushed each other to opposite, extreme coefficients)

                              Rest        Stress      Recovery
    n (ANOCA+ / ANOCA-)       97 (63/34)  97 (63/34)  96 (62/34)
    Global LR test p          0.115       0.049       0.062
    Pseudo R2                 0.16        0.19        0.18
    Hosmer-Lemeshow p         0.094       0.291       0.368
    Biomarkers with p < 0.05  0           1           1

All 14 biomarkers, each adjusted for the other 13:

    Legend: each cell is  OR (95% CI) p
        OR      = odds ratio of ANOCA per 1 SD higher logit (> 1: higher logit goes with ANOCA+, < 1: with ANOCA-)
        95% CI  = 95% confidence interval of the OR (if it does not include 1, p < 0.05)
        p       = Wald p value for that biomarker

    Biomarker                                          Rest                    Stress                  Recovery
                                                       OR (95% CI)      p      OR (95% CI)      p      OR (95% CI)      p
    NONSPECIFIC T WAVE ABNORMALITY                   2.37 (0.86-6.52) 0.096  1.16 (0.51-2.63) 0.732  2.02 (0.83-4.94) 0.122
    WITH REPOLARIZATION ABNORMALITY                    0.81 (0.26-2.49) 0.715  0.71 (0.23-2.17) 0.549  0.84 (0.31-2.23) 0.721
    NONSPECIFIC ST ABNORMALITY                         0.84 (0.44-1.59) 0.583  1.41 (0.71-2.79) 0.331  1.74 (0.88-3.43) 0.111
    EARLY REPOLARIZATION                               0.59 (0.27-1.27) 0.179  0.44 (0.20-0.98) 0.043  0.84 (0.39-1.78) 0.645
    PROLONGED QT                                       0.68 (0.36-1.29) 0.238  0.65 (0.32-1.33) 0.238  0.74 (0.38-1.48) 0.397
    NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY      0.63 (0.35-1.11) 0.109  0.59 (0.31-1.11) 0.104  0.51 (0.26-0.98) 0.044
    RSR' PATTERN IN V1                                 1.02 (0.56-1.87) 0.945  0.86 (0.48-1.57) 0.625  1.08 (0.60-1.94) 0.801
    LOW VOLTAGE QRS                                    1.17 (0.65-2.14) 0.602  0.99 (0.53-1.85) 0.964  1.10 (0.59-2.05) 0.760
    VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY  1.48 (0.50-4.40) 0.484  1.90 (0.64-5.62) 0.246  1.48 (0.55-3.95) 0.436
    WITH 1ST DEGREE AV BLOCK                           0.70 (0.34-1.44) 0.331  0.70 (0.35-1.39) 0.311  0.94 (0.46-1.92) 0.856
    WITH SHORT PR                                      0.84 (0.49-1.44) 0.521  0.66 (0.38-1.16) 0.145  1.15 (0.67-1.97) 0.606
    NO P-WAVES FOUND                                   1.37 (0.66-2.84) 0.394  1.88 (0.94-3.79) 0.076  0.69 (0.31-1.55) 0.372
    LEFT AXIS DEVIATION                                1.34 (0.65-2.73) 0.430  1.68 (0.79-3.61) 0.181  1.45 (0.73-2.90) 0.290
    RIGHT AXIS DEVIATION                               1.26 (0.69-2.28) 0.453  1.39 (0.75-2.58) 0.300  1.30 (0.68-2.47) 0.424

Biomarkers with p < 0.05 (each adjusted for the other 13):

    Phase     Biomarker                                       OR per SD (95% CI)   p
    Stress    EARLY REPOLARIZATION                            0.44 (0.20-0.98)     0.043   same as in the 15 predictor model
    Recovery  NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY   0.51 (0.26-0.98)     0.044   was 0.22 with QRS widening in

In a multivariable model of 14 ECGFounder biomarker logits, the overall association with ANOCA was significant
in stress ECGs (likelihood ratio chi2 = 23.8, df = 14, p = 0.049), borderline in recovery (p = 0.062) and not
significant at rest (p = 0.115). A higher nonspecific intraventricular conduction delay logit went with lower
odds of ANOCA in every phase and reached significance in recovery (OR 0.51 per SD, 95% CI 0.26-0.98, p = 0.044).
A higher early repolarization logit went with lower odds of ANOCA in stress (OR 0.44, 0.20-0.98, p = 0.043).

"""