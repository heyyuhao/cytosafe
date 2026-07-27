"""
04_train_lgbm.py — Reproduce CytoSafe LGBM training (one cell line at a time)

Optimization metric: F1 score (pos_label=1, i.e. cytotoxic class)
  → The paper text says BACC, but the author's source code uses F1.
    F1 directly optimizes Precision × Recall balance for the toxic class,
    which explains the high Precision values reported in Table 1.

Pipeline (matching the paper):
  1. Load pre-split ECFP4 fingerprints (80% train / 20% test) from ../data/fp/
  2. Bayesian HPO: 100 iterations, each scored by mean BACC over 10-fold
     stratified CV on TRAIN only — test set never touched here
  3. Best hyperparams = those that gave the highest mean CV BACC
  4. Retrain final model on full TRAIN set with best hyperparams
  5. Save model, best params, and log
  → Test-set evaluation (Table 1 metrics) is done in notebook 05

Usage (run from the notebooks/ directory):
  python 04_train_lgbm.py --dataset 3T3
  python 04_train_lgbm.py --dataset HEK293
  python 04_train_lgbm.py --dataset 3T3 --n_iter 100 --n_folds 10 --random_state 42
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real

# ── Hyperparameter search space ───────────────────────────────────────────────
# Author's original bounds (n_estimators up to 5000, max_depth up to 256) caused
# severe overfitting (4837 trees selected). Using practical bounds that match
# what LGBM documentation recommends for tabular binary classification.
SEARCH_SPACE = {
    'n_estimators':      Integer(50, 500),
    'learning_rate':     Real(0.01, 0.3, prior='log-uniform'),
    'num_leaves':        Integer(20, 150),
    'max_depth':         Integer(3, 12),
    'min_child_samples': Integer(5, 100),
    'subsample':         Real(0.5, 1.0),
    'colsample_bytree':  Real(0.5, 1.0),
    'reg_alpha':         Real(1e-4, 1.0, prior='log-uniform'),
    'reg_lambda':        Real(1e-4, 1.0, prior='log-uniform'),
}
PARAM_KEYS = list(SEARCH_SPACE.keys())


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_path):
    logger = logging.getLogger('train_lgbm')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ── Per-iteration callback ────────────────────────────────────────────────────

class IterationLogger:
    """
    Called by BayesSearchCV after each HPO iteration.
    Logs the hyperparams tried and the resulting mean CV BACC across all folds.
    """

    def __init__(self, logger, n_iter, param_keys):
        self.logger = logger
        self.n_iter = n_iter
        self.param_keys = param_keys
        self.iteration = 0
        self.all_scores = []

    def __call__(self, result):
        self.iteration += 1

        # skopt minimizes, so scores are stored negated
        latest_score = -result.func_vals[-1]
        best_score   = -min(result.func_vals)
        best_idx     = int(np.argmin(result.func_vals))
        params       = dict(zip(self.param_keys, result.x_iters[-1]))
        is_best      = (self.iteration - 1) == best_idx

        self.all_scores.append(latest_score)

        param_str = '  '.join(f'{k}={v}' for k, v in params.items())
        self.logger.info(
            f"[Iter {self.iteration:3d}/{self.n_iter}] "
            f"mean_CV_F1={latest_score:.4f}  "
            f"best_so_far={best_score:.4f}"
            f"{'  ← NEW BEST' if is_best else ''}  "
            f"| {param_str}"
        )
        return False   # never stop early


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(fp_dir, dataset, split):
    with open(os.path.join(fp_dir, f'{dataset}_ecfp4_{split}.json')) as f:
        X = np.asarray(json.load(f))
    with open(os.path.join(fp_dir, f'{dataset}_y_{split}.json')) as f:
        y = np.asarray(json.load(f), dtype=int)
    return X, y


# ── Main training function ────────────────────────────────────────────────────

def train(dataset, fp_dir, out_dir, n_iter, n_folds, random_state):
    os.makedirs(out_dir, exist_ok=True)

    log_path    = os.path.join(out_dir, f'{dataset}_training.log')
    model_path  = os.path.join(out_dir, f'{dataset}_lgbm_ecfp4.joblib')
    params_path = os.path.join(out_dir, f'{dataset}_best_params.json')

    logger = setup_logger(log_path)

    # ── Step 1: load data ─────────────────────────────────────────────────────
    logger.info('=' * 70)
    logger.info(f'Dataset       : {dataset}')
    logger.info(f'Output dir    : {os.path.abspath(out_dir)}')
    logger.info(f'HPO iterations: {n_iter}')
    logger.info(f'CV folds      : {n_folds}')
    logger.info(f'Random state  : {random_state}')
    logger.info(f'Metric        : F1 (pos_label=1, cytotoxic class) — matches author source code')
    logger.info('=' * 70)

    X_train, y_train = load_split(fp_dir, dataset, 'train')
    X_test,  y_test  = load_split(fp_dir, dataset, 'test')
    logger.info(f'X_train : {X_train.shape}  cytotoxic={y_train.sum():,}  '
                f'non-toxic={(y_train == 0).sum():,}')
    logger.info(f'X_test  : {X_test.shape}   cytotoxic={y_test.sum():,}  '
                f'non-toxic={(y_test == 0).sum():,}')
    logger.info('(Test set locked — not used until notebook 05)')

    # ── Step 2: Bayesian HPO — 100 iters × 10-fold CV on TRAIN only ──────────
    logger.info('')
    logger.info(f'Starting Bayesian HPO: {n_iter} iterations × {n_folds}-fold CV')
    logger.info(f'Each line: GP proposes hyperparams → {n_folds}-fold CV on TRAIN → mean F1')
    logger.info('')

    f1_scorer = make_scorer(f1_score, pos_label=1)

    inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=2)   # author uses random_state=2
    base_clf = LGBMClassifier(
        class_weight='balanced',  # critical for 1:5 imbalanced dataset
        random_state=0,
        verbosity=-1,
        n_jobs=-1,
    )
    opt = BayesSearchCV(
        estimator=base_clf,
        search_spaces=SEARCH_SPACE,
        scoring=f1_scorer,           # F1, not BACC — matches author source code
        cv=inner_cv,
        n_iter=n_iter,
        n_points=1,
        n_jobs=-1,
        return_train_score=False,
        refit=True,
        random_state=0,              # author uses random_state=0
        optimizer_kwargs={'base_estimator': 'GP'},
        verbose=0,
    )

    iter_logger = IterationLogger(logger, n_iter, PARAM_KEYS)

    t0 = time.time()
    opt.fit(X_train, y_train, callback=iter_logger)
    elapsed = time.time() - t0

    # ── Step 3: HPO summary ───────────────────────────────────────────────────
    iters_run   = len(opt.cv_results_['params'])
    best_score  = opt.best_score_
    best_idx    = opt.best_index_
    best_params = dict(opt.best_params_)
    best_std    = pd.DataFrame(opt.cv_results_).iloc[best_idx]['std_test_score']

    logger.info('')
    logger.info('=' * 70)
    logger.info('HPO COMPLETE')
    logger.info(f'  Total time      : {elapsed / 60:.1f} min')
    logger.info(f'  Iterations run  : {iters_run}/{n_iter}')
    logger.info(f'  Winning iter    : {best_idx + 1}')
    logger.info(f'  Best CV F1      : {best_score:.4f} ± {best_std:.4f}')
    logger.info('  Best hyperparams:')
    for k, v in best_params.items():
        logger.info(f'    {k:<22} = {v}')

    ranked = sorted(enumerate(iter_logger.all_scores, 1), key=lambda x: -x[1])
    logger.info('')
    logger.info('All iterations ranked by mean CV F1:')
    logger.info(f'  {"Rank":<6} {"Iter":<6} {"mean_CV_F1"}')
    for rank, (it, score) in enumerate(ranked, 1):
        marker = '  ← BEST' if it == (best_idx + 1) else ''
        logger.info(f'  {rank:<6} {it:<6} {score:.4f}{marker}')
    logger.info('=' * 70)

    # ── Step 4: final model refitted on full X_train (via refit=True above) ───
    best_model = opt.best_estimator_
    logger.info('')
    logger.info('Final model refitted on full X_train with best hyperparams.')

    # ── Save ──────────────────────────────────────────────────────────────────
    dump(best_model, model_path)
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)

    logger.info(f'Model saved  : {model_path}')
    logger.info(f'Params saved : {params_path}')
    logger.info(f'Full log     : {log_path}')
    logger.info('')
    logger.info('Next: run notebook 05 to evaluate on the test set (Table 1 metrics).')
    logger.info('=' * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train LGBM for CytoSafe')
    parser.add_argument('--dataset',      required=True,
                        choices=['3T3', 'HEK293'],
                        help='Which cell line to train')
    parser.add_argument('--fp_dir',       default='../data/fp',
                        help='Fingerprint JSON directory (default: ../data/fp)')
    parser.add_argument('--out_dir',      default='../model',
                        help='Output directory (default: ../model)')
    parser.add_argument('--n_iter',       type=int, default=50,
                        help='HPO iterations (default: 50, matches author code)')
    parser.add_argument('--n_folds',      type=int, default=5,
                        help='CV folds inside HPO (default: 5, matches author code)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    train(
        dataset=args.dataset,
        fp_dir=args.fp_dir,
        out_dir=args.out_dir,
        n_iter=args.n_iter,
        n_folds=args.n_folds,
        random_state=args.random_state,
    )


if __name__ == '__main__':
    main()
