# Next steps: predicting ANOCA from ECGFounder embeddings

So far (LogisticRegressionOnOutputs.py, MannWhitneyOnOutputs.py) the work has been **inference**: which ECG
biomarkers are associated with ANOCA, with p values and odds ratios.

The next step is a **prediction** task: how well can ANOCA be predicted from the ECG, measured on people the model
has not seen. No p values per feature, no interpretation of coefficients; the result is the out-of-fold performance.

## Data

- **Input:** the ECGFounder embeddings (`deep_features`, 1024 values per 10 second slice, the layer before the
  150 class outputs). `generateOutput` in embedding.py already returns them, but `getPersonLogits` only saves the
  logits, so first save the embeddings too (e.g. `csv/ScienceDB_slice_embeddings.npy` + the matching rows of
  `ScienceDB_label_all_phases.csv`).
- **10 second slices:** yes, ECGFounder needs 10 second, 12 lead, 500 Hz input, so each slice gives one embedding.
  But the label (ANOCA) is per person, so the person is the unit of the analysis:
  - **Split folds by patient**, never by slice (`StratifiedGroupKFold` with `groups = patient_id`). If slices of the
    same person are in both train and test, the model partly learns to recognize the person and the AUC is inflated.
  - **Report metrics per person**: either average the slice embeddings per person and phase (like the logits), or
    predict per slice and average the predicted probabilities per person. Start with averaged embeddings (simpler).
- **Phases:** one model per phase (rest, stress, recovery), and optionally one with the three phase embeddings
  of a person concatenated.
- **Size:** ~97 people (63 ANOCA+, 34 ANOCA-) and 1024 features, so overfitting is the main risk.

## Models

The current CV AUC (~0.66) is a lower bound: it is a linear model on the 150 logits, and the logits are only a
linear 150 dimensional projection of the 1024 embeddings (logits = W * embedding + b), so information that does
not line up with one of the 150 labels is lost. Go up a ladder, same folds for every step:

1. **Ridge logistic regression on the 150 logits** (current baseline, ~0.66).
2. **Ridge logistic regression on the 1024 embeddings** (same as `logisticModel()`): do the embeddings add information?
3. **Small MLP head on the embeddings** (1 hidden layer of ~16-64 units + 1 output for ANOCA, dropout,
   weight decay, early stopping inside the training folds): does nonlinearity add anything?
   This is also the model to use for morphing (below), because ECG -> ECGFounder -> MLP is differentiable end to end.
4. **XGBoost** (shallow trees, max_depth 2-3, small learning rate, row / column subsampling) as another nonlinear
   check. Not differentiable, so it cannot be used for morphing.
5. Optional: PCA (10-20 components, fit inside the training folds only) before any of them.

Expectation: with ~97 people (34 ANOCA-) the limit is probably the sample size, not the model. A 32 unit hidden
layer on 1024 inputs has ~33,000 weights, so the training AUC will be near 1 and only the out of fold AUC counts.
The 95% CI of an AUC around 0.7 with 97 people is about +-0.1, so small differences between models are noise.
If the MLP does not clearly beat ridge on the embeddings, more data (e.g. NYU) is the way forward.
Fix the architecture in advance (or use nested CV), since picking the best of many architectures on the same
folds inflates the AUC. Training on slices instead of person averages gives more examples, but slices of one
person are highly correlated, so the effective sample size is still ~97 people and folds stay grouped by patient.

## Validation

- **5 fold stratified, grouped by patient, repeated** (e.g. 10 repeats with different seeds), because a single
  5 fold split on 97 people is noisy.
- **Everything fit inside the training folds**: scaling, PCA, hyperparameter tuning (nested CV, or a small fixed grid),
  and the decision threshold. Nothing may see the test fold.
- **Permutation test**: shuffle the ANOCA labels and repeat (e.g. 200-1000 times) to show the AUC is above chance.

## Metrics to report (out of fold, per person)

- **AUROC** with 95% CI (bootstrap over people, or spread over the repeats). Main metric.
- **AUPRC** (average precision), compared with the ANOCA+ prevalence (~0.65), which is its chance level.
- **Sensitivity, specificity, PPV, NPV, balanced accuracy** at a threshold chosen in the training folds
  (e.g. Youden index), not at 0.5 by default and not tuned on the test fold.
- **Calibration**: Brier score and a calibration plot.
- **Permutation p value** for the AUROC.
- ROC curve (mean over folds) and confusion matrix as figures.
- Optional: SHAP or feature importance only as a description of the model, not as inference.

# Next steps: deterministic ECG biomarker functions

The ECGFounder logits are learned, black box estimates of "this ECG has finding X". Next, write our own
**deterministic, rule based functions** that measure the biomarkers directly on a 10 second slice, so the
results can be explained and checked: e.g. "ST depression of 0.07 mV at J+60 ms in V5 and V6".

## What already exists

- **In this repo:** `ECG_Preprocessing/preprocessingForLogisticRegression.py` already does NeuroKit2 delineation
  and per beat features (T amplitude, QTcF, PR, HR, HRV, R-T amplitude), mostly on lead II, and
  `DelinationValidation.py` validates the delineation against QTDB and LUDB. The new functions build on these.
- **Python packages:** they measure but do not interpret. As far as we know there is no maintained, validated
  Python package that takes a 12 lead slice and says "ST depression: yes / no".
  - NeuroKit2: R peaks, P/QRS/T onsets, peaks and offsets (delineation), HR and HRV. No ST level or T inversion calls.
  - wfdb, BioSPPy, HeartPy: QRS detection and HRV.
  - ECGDeli (MATLAB, not Python): delineation and amplitude / interval features.
  - The interpretive algorithms used clinically (GE Marquette 12SL, Glasgow Uni-G) are proprietary.
    Their outputs are available for PTB-XL through PTB-XL+ (see below), which makes them a good reference.

## Possible path: use ECGDeli without porting it to Python

Porting ECGDeli is a large job (weeks), would need validating again, and (we believe) its GPL license would apply
to the port. Not needed: the cluster has MATLAB (`module avail matlab`: R2014a ... R2026b).

1. **Run ECGDeli in MATLAB batch mode, read the results in Python (preferred).** One small MATLAB script loops over
   the slices, runs ECGDeli and writes the wave boundaries / amplitudes to a csv, then pandas reads it:
   ```bash
   module load matlab/R2024a
   matlab -batch "run_ecgdeli('data/ScienceDB_Data', 'csv/ecgdeli_features.csv')"
   ```
2. **MATLAB Engine API for Python** (`pip install matlabengine==24.1.*` for R2024a, supports Python 3.10):
   call ECGDeli from Python directly. Works, but array conversion and MATLAB start up make it fiddlier.
3. GNU Octave + oct2py: free, but ECGDeli uses toolbox functions Octave does not fully support, and Octave is
   not on the cluster. Skip.

First check that the needed toolboxes (at least Signal Processing Toolbox) are in the NYU license:
`matlab -batch "ver"` on a compute node.

Most of ECGDeli's value is the delineation, which we already have (NeuroKit2, validated on LUDB / QTDB), and the
biomarker rules are short on top of it. So the practical plan:
1. Write the biomarker rules in Python on top of the NeuroKit2 delineation.
2. Run ECGDeli (MATLAB batch) on the same slices as a cross-check.
3. Validate both against PTB-XL+, which already contains ECGDeli's measurements for every PTB-XL ECG.

## Biomarkers to implement (per lead, per beat or on a median beat)

Thresholds should come from published criteria (AHA/ACCF/HRS 2007-2009 ECG standardization recommendations,
Minnesota Code), not chosen by us.

- **ST deviation:** ST level at the J point and J+60 / J+80 ms minus the PR (or TP) baseline, in mV.
  ST depression and ST elevation follow from per lead thresholds in 2 contiguous leads.
- **T wave:** T amplitude (signed), T inversion per lead, flat / low T, T/R ratio.
- **QRS:** QRS duration (widening >= 110-120 ms, conduction delay), QRS voltage (low voltage: < 0.5 mV in all
  limb leads or < 1.0 mV in all precordial leads), LVH voltage criteria (Sokolow-Lyon, Cornell).
- **Intervals:** PR (1st degree AV block > 200 ms, short PR < 120 ms), QT and QTc (Bazett, Fridericia).
- **Axis:** frontal QRS axis from the net QRS area in leads I and aVF (left / right axis deviation).
- **Early repolarization:** J point elevation / notching in inferior or lateral leads.
- **P wave:** presence, duration and amplitude (atrial enlargement criteria).

## Data to validate the functions (like LUDB / QTDB for the delineation)

- **PTB-XL** (already used by `ptbxl_eval.py` on Windows, not yet copied to the cluster): 21,799 10 s 12 lead ECGs with cardiologist statements, including
  ST depression, ST elevation, nonspecific ST changes, inverted / low T, low voltage, IVCD, 1st degree AV block,
  long QT, plus a heart axis column. Use it to check our yes / no calls (sensitivity, specificity, AUC).
- **PTB-XL+** (PhysioNet): measurements for every PTB-XL ECG from GE 12SL, Glasgow Uni-G and ECGDeli, per lead
  (ST amplitudes, T amplitude, QRS duration, PR, QT, axis, ...). The best reference for our *continuous*
  measurements (correlation, Bland-Altman against 12SL / Uni-G).
- **European ST-T Database** (PhysioNet): ambulatory 2 lead records with ST and T episodes annotated beat by beat
  by cardiologists. The classic ST deviation benchmark.
- **Long-Term ST Database** (PhysioNet): longer ambulatory records with annotated ST episodes.
- **PhysioNet / CinC Challenge 2020-2021 data** (Chapman-Shaoxing, Ningbo, Georgia, CPSC, ...): tens of
  thousands of 12 lead ECGs with SNOMED labels such as ST depression, ST elevation, T wave abnormality,
  T inversion, low QRS voltage, prolonged QT, IVCD, 1st degree AV block, left / right axis deviation.
  Good as a second, independent test set.
- **LUDB** and **QTDB** (already used): P / QRS / T boundaries, for the intervals, durations and axis.

## Then

- Run the validated functions on the ScienceDB slices (per phase), average per person, and repeat the
  Mann-Whitney / logistic analyses with these measured biomarkers instead of (or next to) the ECGFounder logits.
- Compare: does the measured biomarker agree with the matching ECGFounder logit (e.g. our ST depression vs the
  REPOLARIZATION ABNORMALITY logit, our QRS duration vs the CONDUCTION DELAY logit)?

# Possible path: discover ANOCA biomarkers by morphing ECGs (Obermeyer et al.)

Paper: Obermeyer, Schubert, Ross et al., "An ECG biomarker for sudden cardiac death discovered with deep
learning", Nature 2026 (doi:10.1038/s41586-026-10674-6, UC Berkeley).
A 64 layer ResNet predicted 1 year sudden cardiac death from ~440k ECGs. To see *what* the model uses (saliency maps
only show *where*), they paired it with a generative ECG model and morphed a low risk patient's ECG step by step
into a high risk version of the same ECG. The morph showed a new feature (slurring in lead aVL), which then became a
measurable, interpretable biomarker.

Idea for us: do the same with an ANOCA predictor, to find which ECG change the model is using, and then turn that
change into one of our deterministic biomarker functions (previous section).

Steps:
1. **A predictor with real signal.** Morphing only explains what the model learned, so it needs a model that
   predicts ANOCA well on unseen people: the MLP head on the ECGFounder embeddings from the prediction task above
   (differentiable, so the morph can follow its gradient; XGBoost cannot be used here). With ~97 people it may be
   weak and partly overfit, so the morphs may show noise. A larger ANOCA / INOCA cohort (e.g. NYU data) would help.
2. **A generative ECG model** (VAE, diffusion or GAN) trained on many 12 lead 10 s ECGs (PTB-XL, MIMIC-IV-ECG,
   CinC Challenge data), so that every morphed ECG still looks like a real ECG.
3. **Morph:** start from an ANOCA- person's ECG, move it in the generative model's latent space in the direction
   that raises the predicted ANOCA probability, decode the ECGs along the way, and look at what changes (which wave,
   which lead) across many people.
4. **Sanity check first:** morph towards an ECGFounder class we understand (e.g. NONSPECIFIC T WAVE ABNORMALITY or
   PROLONGED QT). If the morph does not change the T wave / QT, the pipeline is not working.
5. **Turn the change into a rule** (e.g. "ST slope in V5"), measure it with a deterministic function, and test it
   on people the model never saw. The morph itself is only hypothesis generating; the test on new data is the result.

Caveats: needs a GPU and a generative ECG model (pretrained or trained by us); the generator should be checked for
realistic ECGs (e.g. delineation still works on them, a cardiologist cannot tell them apart from real ones).
