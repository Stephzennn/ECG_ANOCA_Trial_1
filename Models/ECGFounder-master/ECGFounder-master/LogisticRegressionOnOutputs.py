import numpy as np
import pandas as pd
import torch
from pathlib import Path
from net1d import Net1D
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score, permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2, spearmanr
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

# Same for the 1024 embeddings (deep_features, the layer before the 150 logits), averaged per person and phase
personEmbeddingsPath = base_path / "csv" / "ScienceDB_person_embeddings.csv"
embeddingColumns = [f"emb_{i}" for i in range(1024)]

# Which part of the __main__ code to run:
#   runInference:  steps 1-4, which logits are associated with ANOCA (bivariate, multivariable, CV AUC, permutation)
#   runPrediction: the prediction report at the end of the file, 150 logits vs 1024 embeddings
runInference = False
runPrediction = True


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


# Run ECGFounder on every 10 second slice. Returns the dataset (patient and phase of every slice),
# and the 150 logits and 1024 embeddings of every slice, in the same row order.
def runECGFounder():
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
    return testset, all_logits, all_embeddings


# Average the 150 logits per person and phase. If this was already done, just read the saved csv.
def getPersonLogits(rerun = False):
    if personLogitsPath.exists() and rerun == False:
        print(f"Reading saved person logits from {personLogitsPath}")
        return pd.read_csv(personLogitsPath, index_col = ["patient_id", "phase"])

    testset, all_logits, all_embeddings = runECGFounder()
    personTable = averageLogitsPerPerson(testset, all_logits, taskPath)
    personTable.to_csv(personLogitsPath)
    print(f"Saved person logits to {personLogitsPath}")
    return personTable


# Same for the 1024 embeddings (needs one more ECGFounder run the first time, a GPU makes it much faster).
def getPersonEmbeddings(rerun = False):
    if personEmbeddingsPath.exists() and rerun == False:
        print(f"Reading saved person embeddings from {personEmbeddingsPath}")
        return pd.read_csv(personEmbeddingsPath, index_col = ["patient_id", "phase"])

    testset, all_logits, all_embeddings = runECGFounder()
    personTable = averageLogitsPerPerson(testset, all_embeddings, columnNames = embeddingColumns)
    personTable.to_csv(personEmbeddingsPath)
    print(f"Saved person embeddings to {personEmbeddingsPath}")
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


# Nonlinear checks on the same logits: can curved effects or interactions between logits beat the linear model?
# Random forest: no scaling needed. min_samples_leaf = 3 keeps trees from memorising single people (n ~ 97).
def randomForestModel(seed = 0):
    return RandomForestClassifier(n_estimators = 500, max_features = "sqrt", min_samples_leaf = 3, random_state = seed)


# RBF kernel SVM: C picked inside each training fold (like the logistic penalty), gamma = 1 / (n_features * var(X)).
def svmModel():
    return make_pipeline(
        StandardScaler(),
        GridSearchCV(SVC(kernel = "rbf", gamma = "scale"), {"C": [0.01, 0.1, 1, 10, 100]}, cv = 5, scoring = "roc_auc"))


# Gradient boosted trees: each new tree fixes the errors of the ones before. Kept small for n ~ 97:
# depth 2 trees (at most pairwise interactions), slow learning rate, row / column subsampling, L2 on leaf weights.
# Fixed settings, not tuned, so nothing in the test fold helps choose them. n_jobs = 1: the folds already run in parallel.
def xgboostModel(seed = 0):
    return XGBClassifier(
        n_estimators = 200, max_depth = 2, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.5,
        min_child_weight = 3, reg_lambda = 1.0, eval_metric = "logloss", n_jobs = 1, random_state = seed)


# Same seed -> same folds for every call, so the AUCs of two models can be compared fold by fold.
# model: logisticModel() by default, or randomForestModel() / svmModel() / xgboostModel().
def cvLogisticAUC(table, label = "ANOCA", tasks = None, n_splits = 5, n_repeats = 10, seed = 0, model = None):
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    X = table[tasks].to_numpy()
    y = table[label].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = seed)
    if model is None:
        model = logisticModel()
    return cross_val_score(model, X, y, cv = cv, scoring = "roc_auc", n_jobs = -1)


# Is the cross validated AUC better than chance? Shuffle the labels and refit n_permutations times.
def permutationPValue(table, label = "ANOCA", tasks = None, n_permutations = 200, seed = 0, model = None):
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    X = table[tasks].to_numpy()
    y = table[label].to_numpy()
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    if model is None:
        model = logisticModel()
    score, permScores, pValue = permutation_test_score(
        model, X, y, cv = cv, scoring = "roc_auc", n_permutations = n_permutations, n_jobs = -1, random_state = seed)
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


if __name__ == "__main__" and runInference:

    personTable = getPersonLogits()

    # quick checks: should be 97 people x 3 phases, 150 class columns + n_ECGs + labels
    print(personTable.shape)
    print(personTable.groupby("phase").size())
    print(personTable[["n_ECGs", "ANOCA", "CMD"]].head())

    # one phase at a time: "rest", "stress" or "recovery" (titles and file names follow this)
    phase = "stress"
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

    # 3b) nonlinear models on all 150 logits, same 50 folds and same permutation test.
    # If they don't beat the logistic AUC, the linear readout is enough.
    for name, makeModel in [("Random forest", randomForestModel), ("RBF SVM", svmModel), ("XGBoost", xgboostModel)]:
        nonlinearAUCs = cvLogisticAUC(phaseTable, label, model = makeModel())
        print(f"\n{name}, all 150 logits: CV AUC = {nonlinearAUCs.mean():.3f} "
              f"(SD {nonlinearAUCs.std():.3f}, {len(nonlinearAUCs)} folds)")

        score, permScores, pValue = permutationPValue(phaseTable, label, model = makeModel())
        print(f"Permutation test: AUC = {score:.3f}, chance AUC = {permScores.mean():.3f}, p = {pValue:.3f}")

        # fold by fold comparison (same 50 folds as the logistic model)
        difference = nonlinearAUCs - foldAUCs
        print(f"{name} minus logistic: mean AUC difference = {difference.mean():+.3f}, "
              f"{name} better in {(difference > 0).sum()} of {len(difference)} folds")

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
Logistic regression, all 150 logits:

                CV AUC, 50 folds    Permutation test (one 5 fold split, 200 shuffles)
                                    AUC     chance AUC   p
    Rest        0.643 (SD 0.104)    0.655   0.502        0.030
    Stress      0.685 (SD 0.103)    0.675   0.497        0.020
    Recovery    0.659 (SD 0.119)    0.717   0.494        0.015

    Report the 50 fold CV AUC; the permutation AUC is from a single split, so it differs by about ±0.05.
    Use the permutation test only for its p value.

Nonlinear check, stress, all 150 logits (same 50 folds and permutation test as the logistic model):

                    CV AUC, 50 folds    Permutation p   vs logistic (mean diff, folds better)
    Logistic        0.685 (SD 0.103)    0.020
    Random forest   0.666 (SD 0.097)    0.030           -0.020, 22 of 50
    RBF SVM         0.666 (SD 0.113)    0.030           -0.019, 21 of 50
    XGBoost         0.690 (SD 0.100)    0.030           +0.004, 28 of 50

    No nonlinear model clearly beat the logistic regression (XGBoost +0.004 is far inside the fold SD of ~0.10),
    so a linear readout of the logits is enough.

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
        *       = p < 0.05

    Biomarker                                          Rest                    Stress                  Recovery
                                                       OR (95% CI)      p      OR (95% CI)      p      OR (95% CI)      p
    NONSPECIFIC T WAVE ABNORMALITY                     2.37 (0.86-6.52) 0.096  1.16 (0.51-2.63) 0.732  2.02 (0.83-4.94) 0.122
    WITH REPOLARIZATION ABNORMALITY                    0.81 (0.26-2.49) 0.715  0.71 (0.23-2.17) 0.549  0.84 (0.31-2.23) 0.721
    NONSPECIFIC ST ABNORMALITY                         0.84 (0.44-1.59) 0.583  1.41 (0.71-2.79) 0.331  1.74 (0.88-3.43) 0.111
    EARLY REPOLARIZATION                               0.59 (0.27-1.27) 0.179  0.44 (0.20-0.98) 0.043* 0.84 (0.39-1.78) 0.645
    PROLONGED QT                                       0.68 (0.36-1.29) 0.238  0.65 (0.32-1.33) 0.238  0.74 (0.38-1.48) 0.397
    NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY      0.63 (0.35-1.11) 0.109  0.59 (0.31-1.11) 0.104  0.51 (0.26-0.98) 0.044*
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

"""
Are the ECGFounder output classes orthogonal? No (one shared backbone, and ECG findings overlap clinically).

All 150 classes (person logits, Spearman r):
                                       Rest   Stress  Recovery
    Mean |r| between classes           0.31   0.27    0.27
    Pairs with |r| > 0.7               359    259     271
    PCA components for 90% variance    19     19      19

The 14 biomarkers:
                                       Rest   Stress  Recovery
    Mean |r|                           0.20   0.20    0.20
    Max |r|                            0.73   0.70    0.68
    Pairs with |r| > 0.5 (of 91)       2      2       3
    Highest VIF (REPOLARIZATION ABN.)  5.4    5.4     4.5

    VIF = variance inflation factor, how much collinearity inflates a coefficient's variance (warning at 5-10).
    REPOLARIZATION ABNORMALITY ~5: its sign flipped in every multivariable model, so don't interpret it.
    LVH voltage ~4 and T wave abnormality ~3: moderate, explains their wide CIs. All others < 3.
    The main findings (conduction delay, early repolarization) are on low VIF biomarkers.

ECGFounder output logits are correlated by design (150 classes, about 19 effective dimensions); we reduced
this to 14 clinically distinct biomarkers (maximum |r| = 0.73, VIF <= 5.4).

"""


# ---------------------------------------------------------------------------
# Prediction: how well can ANOCA be predicted for people the model has not seen? (see NEXT_STEPS.md)
# Same model (logisticModel(), ridge with the penalty tuned inside each training fold) and the same 50 folds
# as above, on the 150 logits and on the 1024 embeddings. The result is the out of fold performance, per person.
# ---------------------------------------------------------------------------

# One fold: fit on the training people, predict the test people.
# The threshold is the Youden index (max sensitivity + specificity - 1) of an inner 5 fold CV on the training people
# only, so the test fold never helps choose it (0.5 is not a sensible default with 65% ANOCA+).
def predictFold(model, X, y, train, test, seed = 0):
    inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    innerProb = cross_val_predict(clone(model), X[train], y[train], cv = inner, method = "predict_proba")[:, 1]
    fpr, tpr, thresholds = roc_curve(y[train], innerProb)
    threshold = thresholds[np.argmax(tpr - fpr)]

    foldModel = clone(model).fit(X[train], y[train])
    prob = foldModel.predict_proba(X[test])[:, 1]
    return test, prob, prob >= threshold


# Out of fold predicted probability (and yes / no call) of every person, once per repeat.
# Same seed -> same folds as cvLogisticAUC, so the two inputs are tested on exactly the same people.
# Returns y, prob and predicted, the last two of shape (n_repeats, n_people).
def outOfFoldPredictions(table, label = "ANOCA", tasks = None, n_splits = 5, n_repeats = 10, seed = 0, model = None):
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    if model is None:
        model = logisticModel()
    X = table[tasks].to_numpy()
    y = table[label].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = seed)
    folds = Parallel(n_jobs = -1)(delayed(predictFold)(model, X, y, train, test, seed) for train, test in cv.split(X, y))

    prob = np.full((n_repeats, len(y)), np.nan)
    predicted = np.zeros((n_repeats, len(y)), dtype = bool)
    for i, (test, foldProb, foldPredicted) in enumerate(folds):
        prob[i // n_splits, test] = foldProb          # folds come in order: 5 per repeat
        predicted[i // n_splits, test] = foldPredicted
    return y, prob, predicted


# Metrics of the out of fold predictions, once per repeat (each person predicted once per repeat),
# and the 95% CI of the AUROC: bootstrap over people of the predicted probability averaged over the repeats.
# AUPRC is compared with the ANOCA+ prevalence (its chance level), not with 0.5.
# ANOCA+ is the majority (~65%), so AUPRC for ANOCA+ starts high by chance. AUPRC_negative treats ANOCA- (the minority,
# ~35%) as the class to find, with 1 - p as its score: the harder and more informative direction here.
def predictionMetrics(y, prob, predicted, n_bootstrap = 2000, seed = 0):
    rows = []
    for p, yhat in zip(prob, predicted):
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels = [0, 1]).ravel()
        rows.append({
            "AUROC": roc_auc_score(y, p),
            "AUPRC": average_precision_score(y, p),
            "AUPRC_negative": average_precision_score(1 - y, 1 - p),
            "Brier": brier_score_loss(y, p),
            "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "PPV": tp / (tp + fp) if tp + fp else np.nan,
            "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "Balanced accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    perRepeat = pd.DataFrame(rows)

    meanProb = prob.mean(axis = 0)
    rng = np.random.default_rng(seed)
    bootAUCs = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(y), len(y))
        if y[sample].min() != y[sample].max():      # needs both classes
            bootAUCs.append(roc_auc_score(y[sample], meanProb[sample]))
    aucCI = np.percentile(bootAUCs, [2.5, 97.5])
    return perRepeat, meanProb, aucCI


# ROC curve and calibration plot of the out of fold probabilities (averaged over the repeats), one line per input.
def plotPrediction(y, meanProbs, labelName = "ANOCA", title = "", outputPath = None):
    fig, (axROC, axCal) = plt.subplots(1, 2, figsize = (11, 5))
    for name, meanProb in meanProbs.items():
        fpr, tpr, _ = roc_curve(y, meanProb)
        axROC.plot(fpr, tpr, lw = 1.8, label = f"{name} (AUROC {roc_auc_score(y, meanProb):.2f})")
        # 5 bins of equal size: with ~97 people, more bins are mostly noise
        fracPositive, meanPredicted = calibration_curve(y, meanProb, n_bins = 5, strategy = "quantile")
        axCal.plot(meanPredicted, fracPositive, "o-", lw = 1.8, label = f"{name} (Brier {brier_score_loss(y, meanProb):.3f})")

    axROC.plot([0, 1], [0, 1], "k--", lw = 1)
    axROC.set_xlabel("1 - specificity")
    axROC.set_ylabel("Sensitivity")
    axROC.set_title("ROC, out of fold")
    axROC.legend(loc = "lower right")
    axCal.plot([0, 1], [0, 1], "k--", lw = 1)
    axCal.set_xlabel(f"Predicted probability of {labelName}+")
    axCal.set_ylabel(f"Observed fraction {labelName}+")
    axCal.set_title("Calibration, out of fold (5 quantile bins)")
    axCal.legend(loc = "upper left")
    fig.suptitle(title)
    fig.tight_layout()
    if outputPath:
        fig.savefig(outputPath, dpi = 200, bbox_inches = "tight")
        print(f"Saved {outputPath}")
    plt.show()


# Which of the 150 classes does XGBoost use? One XGBoost model per fold (same 50 folds as above), and for the
# test people of that fold the SHAP value of every class: how much that class moved this person's predicted
# log odds of ANOCA+ away from the average (TreeSHAP, built into xgboost as pred_contribs, no shap package needed).
# Computed on people the fold model did not see, so it describes what generalises, not what was memorised.
#   mean |SHAP|: overall importance (log odds units), averaged over all people and repeats
#   direction:   Spearman r between the class logit and its SHAP value, > 0: higher logit pushes towards ANOCA+
#   gain:        xgboost's own importance (average loss reduction of the splits on that class), mean over the 50 models
# Description of the model only, not inference: no p values, and correlated classes share (or steal) importance.
def xgboostFold(X, y, train, test, seed = 0):
    foldModel = xgboostModel(seed).fit(X[train], y[train])
    booster = foldModel.get_booster()
    contribs = booster.predict(xgb.DMatrix(X[test]), pred_contribs = True)[:, :-1]     # last column is the bias
    gain = booster.get_score(importance_type = "gain")                                  # keys f0, f1, ...
    return test, contribs, np.array([gain.get(f"f{j}", 0.0) for j in range(X.shape[1])])


def xgboostImportance(table, label = "ANOCA", tasks = None, n_splits = 5, n_repeats = 10, seed = 0):
    if tasks is None:
        tasks = loadTaskNames(taskPath)
    X = table[tasks].to_numpy()
    y = table[label].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = seed)
    folds = Parallel(n_jobs = -1)(delayed(xgboostFold)(X, y, train, test, seed) for train, test in cv.split(X, y))

    # every person appears once per repeat, so these are n_repeats x n_people rows
    shap = np.concatenate([contribs for test, contribs, gain in folds])
    values = np.concatenate([X[test] for test, contribs, gain in folds])
    result = pd.DataFrame({
        "mean_abs_SHAP": np.abs(shap).mean(axis = 0),
        "direction_r": [spearmanr(values[:, j], shap[:, j])[0] if shap[:, j].any() else np.nan for j in range(len(tasks))],
        "gain": np.mean([gain for test, contribs, gain in folds], axis = 0),
        "used_in_folds": np.mean([gain > 0 for test, contribs, gain in folds], axis = 0)}, index = tasks)
    return result.sort_values("mean_abs_SHAP", ascending = False)


# Kernel density of the out of fold predicted probability, ANOCA- vs ANOCA+ (like KernelDensityPlot.py).
# The more the two curves are apart, the better the model separates the groups; the overlap is what the AUROC measures.
# Ticks under the curves are the individual people.
def plotPredictionDensity(y, prob, labelName = "ANOCA", title = "", outputPath = None):
    probs = pd.Series(prob)
    fig, ax = plt.subplots(figsize = (8, 5))
    for value, color, name in [(0, "steelblue", f"{labelName}- (n={(y == 0).sum()})"), (1, "darkorange", f"{labelName}+ (n={(y == 1).sum()})")]:
        probs[y == value].plot.kde(ax = ax, color = color, linewidth = 2.5, label = name)
        ax.plot(probs[y == value], np.full((y == value).sum(), -0.05 - 0.1 * value), "|", color = color, markersize = 12)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Out of fold predicted probability of {labelName}+")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nAUROC {roc_auc_score(y, prob):.2f}")
    ax.grid(True, linestyle = "--", alpha = 0.6)
    ax.legend()
    fig.tight_layout()
    if outputPath:
        fig.savefig(outputPath, dpi = 200, bbox_inches = "tight")
        print(f"Saved {outputPath}")
    plt.show()


if __name__ == "__main__" and runPrediction:

    phase = "stress"
    label = "ANOCA"

    logitTable = getPersonLogits().xs(phase, level = "phase")
    embeddingTable = getPersonEmbeddings().xs(phase, level = "phase")

    # same people in the same order, otherwise the folds would not match
    assert logitTable.index.equals(embeddingTable.index)
    assert (logitTable[label] == embeddingTable[label]).all()
    print(f"\nPrediction, {phase}, {label}: {logitTable[label].sum()} positive, {(logitTable[label] == 0).sum()} negative")

    # every input x model combination, all on the same 50 folds.
    # Ridge logistic = linear probe; XGBoost = shallow trees, can use curved effects and pairwise interactions.
    inputs = {"150 logits": (logitTable, loadTaskNames(taskPath)), "1024 embeddings": (embeddingTable, embeddingColumns)}
    models = {"ridge": logisticModel, "XGBoost": xgboostModel}
    meanProbs, foldAUCs = {}, {}
    for inputName, (table, tasks) in inputs.items():
        for modelName, makeModel in models.items():
            name = f"{inputName}, {modelName}"
            y, prob, predicted = outOfFoldPredictions(table, label, tasks, model = makeModel())
            perRepeat, meanProbs[name], aucCI = predictionMetrics(y, prob, predicted)

            print(f"\n{name}, out of fold over 10 repeats of 5 fold CV:")
            print(f"    AUROC {perRepeat['AUROC'].mean():.3f} (95% CI {aucCI[0]:.2f}-{aucCI[1]:.2f}, bootstrap over people)")
            print(f"    AUPRC {perRepeat['AUPRC'].mean():.3f} (chance = prevalence {y.mean():.2f})")
            print(f"    AUPRC for {label}- {perRepeat['AUPRC_negative'].mean():.3f} (chance = prevalence {1 - y.mean():.2f})")
            print(f"    Brier {perRepeat['Brier'].mean():.3f} (always predicting the prevalence: {y.mean() * (1 - y.mean()):.3f})")
            print("    At the Youden threshold of each training fold (mean over repeats, min-max):")
            for metric in ["Sensitivity", "Specificity", "PPV", "NPV", "Balanced accuracy"]:
                print(f"        {metric:18s} {perRepeat[metric].mean():.2f} ({perRepeat[metric].min():.2f}-{perRepeat[metric].max():.2f})")
            counts = perRepeat[["TP", "FP", "FN", "TN"]].mean()
            print(f"    Confusion matrix (mean people per repeat): predicted ANOCA+: {counts.TP:.1f} true, {counts.FP:.1f} false; "
                  f"predicted ANOCA-: {counts.TN:.1f} true, {counts.FN:.1f} false")

            # fold AUCs (same 50 folds for every combination) and the permutation test, as in steps 2-3
            foldAUCs[name] = cvLogisticAUC(table, label, tasks, model = makeModel())
            print(f"    Mean fold AUC {foldAUCs[name].mean():.3f} (SD {foldAUCs[name].std():.3f}, {len(foldAUCs[name])} folds)")
            score, permScores, pValue = permutationPValue(table, label, tasks, model = makeModel())
            print(f"    Permutation test: AUC = {score:.3f}, chance AUC = {permScores.mean():.3f}, p = {pValue:.3f}")

    # fold by fold, everything against the linear probe on the 150 logits
    baseline = "150 logits, ridge"
    print(f"\nFold by fold against {baseline} (same 50 folds):")
    for name in foldAUCs:
        if name != baseline:
            difference = foldAUCs[name] - foldAUCs[baseline]
            print(f"    {name:26s} mean AUC difference = {difference.mean():+.3f}, better in {(difference > 0).sum()} of {len(difference)} folds")

    plotPrediction(
        y, meanProbs, label,
        title = f"{phase.capitalize()} ECG: predicting {label} ({y.sum()}+ / {(y == 0).sum()}-), "
                f"ridge logistic regression and XGBoost, out of fold (10 x 5 fold CV)",
        outputPath = base_path / "Images" / f"prediction_{phase}_{label}.png")

    # kernel density of one model (set bestModel to any of the four names above): how far apart are label+ and label-?
    # Probability averaged over the 10 repeats, each from a model that did not see that person.
    bestModel = "150 logits, ridge"
    plotPredictionDensity(
        y, meanProbs[bestModel], label,
        title = f"{phase.capitalize()} ECG, {bestModel}: out of fold predicted probability (mean of 10 repeats)",
        outputPath = base_path / "Images" / f"prediction_density_{phase}_{label}.png")

    # which of the 150 classes XGBoost uses (out of fold SHAP, same 50 folds)
    importance = xgboostImportance(logitTable, label)
    importancePath = base_path / "csv" / f"ScienceDB_xgboost_importance_{phase}_{label}.csv"
    importance.to_csv(importancePath)
    print(f"\nXGBoost on the 150 logits, top 20 classes by mean |SHAP| (full table saved to {importancePath}):")
    print("    direction_r > 0: higher logit pushes towards ANOCA+; used_in_folds: share of the 50 models that split on it")
    print(importance.head(20).round(3).to_string())

"""
Prediction of ANOCA (63+ / 34-, recovery 62+ / 34-): 150 logits vs 1024 embeddings, ridge and XGBoost,
out of fold over 10 x 5 fold CV (same folds for every model). AUROC and Brier from the figures
Images/prediction_{phase}_ANOCA.png (AUROC of the probability averaged over the 10 repeats, so slightly optimistic).

                    150 logits              1024 embeddings
                    ridge      XGBoost      ridge      XGBoost
    AUROC
    Rest            0.65       0.61         0.67       0.68
    Stress          0.71       0.70         0.75       0.65
    Recovery        0.68       0.67         0.73       0.63
    Brier (lower is better; always predicting the prevalence: 0.226)
    Rest            0.222      0.244        0.214      0.214
    Stress          0.211      0.210        0.196      0.227
    Recovery        0.215      0.222        0.199      0.224

    150 logits, ridge, mean fold AUC and permutation p (from steps 2-3 above):
    rest 0.643 (p 0.030), stress 0.685 (p 0.020), recovery 0.659 (p 0.015)

What it shows:
    - Ridge on the embeddings is the highest in stress and recovery and has the lowest Brier score in all phases.
      Stress is the best phase (0.75).
    - XGBoost never clearly beats ridge, and on the embeddings it is the worst model in stress and recovery
      (shallow trees on 1024 features with ~77 training people overfit). A linear readout is enough.
    - Stress, ridge on the embeddings (prediction_density_stress_ANOCA.png): most ANOCA+ sit at 0.75-0.95, but
      about half of the ANOCA- sit at 0.6-0.75 among them, and ~a dozen ANOCA+ sit at 0.2-0.5. The model separates
      a subgroup, not the whole cohort.

150 logits vs 1024 embeddings:
    - The embeddings are ahead by 0.02-0.05 AUROC (rest +0.02, stress +0.04, recovery +0.05).
    - The 95% CI of an AUROC around 0.7 with 63 / 34 people is about +-0.10 (Hanley-McNeil SE ~0.05), so each
      model's CI covers the other model's AUROC in every phase. The gap is well inside the noise and has not been
      shown to be real (that needs a paired test on the same people, e.g. a paired bootstrap of the AUROC difference
      or the fold by fold wins, ~25 of 50 = a tie).
    - The 150 logits are interpretable: every input is a named ECG finding (T wave abnormality, conduction delay,
      early repolarization, ...), so the model can be explained, checked against the ECG and turned into measurable
      biomarkers (deterministic functions in NEXT_STEPS.md). The 1024 embedding dimensions have no meaning on their own.
    - So unless the embeddings perform markedly better (a paired CI of the difference that excludes 0, and a gain
      that is clinically relevant, not 0.02-0.05), the 150 logits are the better path: almost the same performance,
      and the model can be explained. The embeddings stay as a sensitivity analysis / upper bound, and are revisited
      with more data (NYU), where a real gap would show up with narrower CIs.
    - Picking the best of 4 models x 3 phases on the same 97 people inflates the top AUROC, so the primary model
      (150 logits, ridge) is fixed now, before more comparisons.
"""
