import argparse
from pathlib import Path
import re

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess

from spatial import config, history


FEATURES = ['pet', 'precip', 'forest', 'water', 'temp', 'grass', 'gdp', 'crop', 'built', 'pop', 'elevation', 'slope']


def model(name, trees, seed):
    if name == 'lgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=trees, num_leaves=15, min_child_samples=10, learning_rate=0.05, random_state=seed, n_jobs=1, verbosity=-1)
    if name == 'rf':
        return RandomForestRegressor(n_estimators=trees, min_samples_leaf=3, random_state=seed, n_jobs=1)
    if name == 'dt':
        return DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=seed)
    if name == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=trees, max_depth=4, learning_rate=0.05, random_state=seed, n_jobs=1)
    if name == 'cat':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=trees, depth=4, learning_rate=0.05, random_seed=seed, verbose=False, allow_writing_files=False, thread_count=1)
    raise ValueError('Unknown model')


def split(indices, fraction, seed, groups=None):
    if groups is None:
        return train_test_split(indices, test_size=fraction, random_state=seed)
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    left, right = next(splitter.split(indices, groups=groups[indices]))
    return indices[left], indices[right]


def score(y, predicted):
    return {'r2': float(r2_score(y, predicted)), 'rmse': float(np.sqrt(mean_squared_error(y, predicted))), 'mae': float(mean_absolute_error(y, predicted))}


def analyze(frame, target, args):
    folder = args.output / target
    folder.mkdir(parents=True, exist_ok=True)
    y = pd.to_numeric(frame[target], errors='coerce').replace([np.inf, -np.inf, -9999], np.nan)
    valid = y.notna()
    x = frame.loc[valid, args.features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf, -9999], np.nan)
    y = y[valid].to_numpy()
    row_ids = np.flatnonzero(valid)
    if len(y) < 30:
        raise ValueError('At least 30 observed target values are required')
    groups = None if args.group is None else frame.loc[valid, args.group].to_numpy()
    if groups is not None and pd.isna(groups).any():
        raise ValueError('Group identifiers must not be missing')
    development, test = split(np.arange(len(y)), 0.2, args.seed, groups)
    train, validation = split(development, 0.2, args.seed + 1, groups)
    if min(len(train), len(validation), len(test)) < 2:
        raise ValueError('A split contains fewer than two samples')
    if x.iloc[train].isna().all().any():
        raise ValueError('A feature has no observations in the training split')
    imputer = SimpleImputer(strategy='median')
    imputer.fit(x.iloc[train])
    data = pd.DataFrame(imputer.transform(x), columns=args.features)
    records, candidates = [], {}
    for name in args.models:
        fitted = model(name, args.trees, args.seed)
        fitted.fit(data.iloc[train], y[train])
        candidates[name] = fitted
        for label, ids in [('train', train), ('validation', validation)]:
            records.append({'model': name, 'split': label, **score(y[ids], fitted.predict(data.iloc[ids]))})
    selected = min((r for r in records if r['split'] == 'validation'), key=lambda r: r['rmse'])['model']
    fitted = candidates[selected]
    predicted = fitted.predict(data.iloc[test])
    records.append({'model': selected, 'split': 'test', **score(y[test], predicted)})
    pd.DataFrame(records).to_csv(folder / 'metrics.csv', index=False)
    pd.DataFrame({'row': row_ids[test], 'observed': y[test], 'predicted': predicted}).to_csv(folder / 'test.csv', index=False)
    split_rows = [{'row': int(row_ids[i]), 'split': label} for label, ids in [('train', train), ('validation', validation), ('test', test)] for i in ids]
    pd.DataFrame(split_rows).to_csv(folder / 'split.csv', index=False)
    joblib.dump({'model': fitted, 'imputer': imputer, 'features': args.features, 'target': target}, folder / 'model.joblib')
    ids = np.random.default_rng(args.seed).choice(test, min(len(test), args.samples), replace=False)
    sample = data.iloc[ids]
    explainer = shap.TreeExplainer(fitted)
    values = np.asarray(explainer.shap_values(sample))
    if values.shape != sample.shape:
        raise ValueError('Unexpected SHAP output shape')
    importance = pd.Series(np.abs(values).mean(axis=0), index=args.features).sort_values(ascending=False)
    importance.rename_axis('feature').rename('mean_abs_shap').to_csv(folder / 'importance.csv')
    pd.DataFrame(values, columns=args.features).to_csv(folder / 'shap.csv', index=False)
    sample.to_csv(folder / 'features.csv', index=False)
    pd.DataFrame({'row': row_ids[ids], 'base_value': float(np.asarray(explainer.expected_value).reshape(-1)[0]), 'predicted': fitted.predict(sample)}).to_csv(folder / 'base.csv', index=False)
    curves = []
    for feature in importance.index[:args.top]:
        column = args.features.index(feature)
        xv, sv = sample[feature].to_numpy(), values[:, column]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(xv, sv, s=8, alpha=0.45)
        if np.unique(xv).size >= 4:
            smooth = lowess(sv, xv, frac=0.4, it=2, return_sorted=True)
            smooth = pd.DataFrame(smooth, columns=['value', 'shap']).groupby('value', as_index=False).mean()
            ax.plot(smooth.value, smooth.shap)
            curves.extend({'feature': feature, 'value': r.value, 'shap': r.shap} for r in smooth.itertuples())
        ax.axhline(0, linewidth=0.6)
        ax.set(xlabel=feature, ylabel='SHAP value', title=target)
        fig.tight_layout()
        fig.savefig(folder / f'{feature}.png', dpi=150)
        plt.close(fig)
    pd.DataFrame(curves, columns=['feature', 'value', 'shap']).to_csv(folder / 'curves.csv', index=False)


def sample_main(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).name} sample')
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--size', type=int, default=10000)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.size, args.block) < 1:
        raise ValueError('Sample and block sizes must be positive')
    cfg = config(args.config)
    if args.year not in cfg['years']:
        raise ValueError('The year is not in the historical configuration')
    cfg['years'] = [args.year]
    x, y, mask, _ = history(cfg)
    valid = mask & np.isfinite(x[0]).all(axis=0) & np.isfinite(y[0]).all(axis=0)
    rows, cols = np.where(valid)
    if not len(rows):
        raise ValueError('No complete samples')
    choice = np.random.default_rng(args.seed).choice(len(rows), min(args.size, len(rows)), replace=False)
    rows, cols = rows[choice], cols[choice]
    output = pd.DataFrame({name: x[0, index, rows, cols] for index, name in enumerate(cfg['features'])})
    output['npp'], output['uhi'] = y[0, 0, rows, cols], y[0, 1, rows, cols]
    output['row'], output['col'] = rows, cols
    output['group'] = (rows // args.block) * int(np.ceil(mask.shape[1] / args.block)) + cols // args.block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


def fit_main(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).name} fit')
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--targets', nargs='+', default=['npp', 'uhi'])
    parser.add_argument('--features', nargs='+', default=FEATURES)
    parser.add_argument('--models', nargs='+', choices=['lgbm', 'rf', 'dt', 'xgb', 'cat'], default=['lgbm'])
    parser.add_argument('--group')
    parser.add_argument('--trees', type=int, default=200)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--top', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    names = args.features + args.targets
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', n) for n in names):
        raise ValueError('Features and targets must be distinct ASCII identifiers')
    if args.group in names or min(args.trees, args.samples, args.top) < 1:
        raise ValueError('Invalid group or model settings')
    frame = pd.read_csv(args.csv)
    required = names + ([args.group] if args.group else [])
    if not set(required).issubset(frame.columns):
        raise ValueError('The input table is missing required columns')
    for target in args.targets:
        analyze(frame, target, args)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = {'sample': sample_main, 'fit': fit_main}
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in commands:
        subparsers.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    commands[args.command](remaining)


if __name__ == '__main__':
    main()
