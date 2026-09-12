import argparse
import math
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spatial import config, history, read, write


def scale(values, indices):
    sample = values[indices].astype('float64')
    axes = (0, 2, 3)
    valid = np.isfinite(sample)
    count = valid.sum(axis=axes)
    if np.any(count == 0):
        raise ValueError('A channel has no training observations')
    total = np.where(valid, sample, 0).sum(axis=axes)
    mean = total / count
    delta = sample - mean[None, :, None, None]
    deviation = np.sqrt(np.where(valid, delta ** 2, 0).sum(axis=axes) / count)
    deviation = np.where(deviation > 1e-8, deviation, 1)
    return mean.astype('float32'), deviation.astype('float32')


def normalize(values, mean, deviation):
    return ((values - mean[None, :, None, None]) / deviation[None, :, None, None]).astype('float32')


class Tiles(Dataset):
    def __init__(self, x, y, endpoints, window, patch):
        if window < 1 or patch < 1:
            raise ValueError('Window and patch sizes must be positive')
        self.x, self.y, self.window, self.patch = x, y, window, patch
        self.items = []
        self.masks = {}
        for endpoint in endpoints:
            endpoint = int(endpoint)
            if endpoint < window - 1:
                continue
            inputs_valid = np.isfinite(x[endpoint - window + 1:endpoint + 1]).all(axis=(0, 1))
            valid = np.isfinite(y[endpoint]) & inputs_valid[None]
            self.masks[endpoint] = valid
            for top in range(0, x.shape[2], patch):
                for left in range(0, x.shape[3], patch):
                    if valid[:, top:top + patch, left:left + patch].any():
                        self.items.append((endpoint, top, left))
        if not self.items:
            raise ValueError('No valid samples for this split')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        endpoint, top, left = self.items[index]
        size = self.patch
        height, width = min(size, self.x.shape[2] - top), min(size, self.x.shape[3] - left)
        x = np.zeros((self.window, self.x.shape[1], size, size), dtype='float32')
        y = np.zeros((2, size, size), dtype='float32')
        mask = np.zeros((2, size, size), dtype=bool)
        block = self.x[endpoint - self.window + 1:endpoint + 1, :, top:top + height, left:left + width]
        x[:, :, :height, :width] = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        target = self.y[endpoint, :, top:top + height, left:left + width]
        y[:, :height, :width] = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        mask[:, :height, :width] = self.masks[endpoint][:, top:top + height, left:left + width]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def split_years(years, train_end, validation_end, window):
    years = np.asarray(years)
    if not 1 <= window <= len(years):
        raise ValueError('Invalid sequence window')
    if not years[window - 1] <= train_end < validation_end < years[-1]:
        raise ValueError('Training, validation, and test periods must all be nonempty')
    indices = np.arange(len(years))
    eligible = indices >= window - 1
    return {
        'train': indices[eligible & (years <= train_end)],
        'validation': indices[eligible & (years > train_end) & (years <= validation_end)],
        'test': indices[eligible & (years > validation_end)]
    }


class Cell(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(inputs + hidden, hidden * 4, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c


class ConvLSTM(nn.Module):
    def __init__(self, inputs, hidden=32, outputs=2):
        super().__init__()
        second = max(hidden // 2, 1)
        self.cells = nn.ModuleList([Cell(inputs, hidden), Cell(hidden, second)])
        self.head = nn.Conv2d(second, outputs, 1)

    def forward(self, x):
        b, steps, _, h, w = x.shape
        states = [(x.new_zeros(b, cell.hidden, h, w), x.new_zeros(b, cell.hidden, h, w)) for cell in self.cells]
        result = []
        for step in range(steps):
            value = x[:, step]
            for index, cell in enumerate(self.cells):
                states[index] = cell(value, states[index])
                value = states[index][0]
            result.append(self.head(value))
        return torch.stack(result, dim=1)


class PixelLSTM(nn.Module):
    def __init__(self, inputs, hidden=32, outputs=2):
        super().__init__()
        self.rnn = nn.LSTM(inputs, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        b, t, c, h, w = x.shape
        sequence = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        values = [self.head(self.rnn(chunk)[0]) for chunk in sequence.split(4096)]
        return torch.cat(values).reshape(b, h, w, t, -1).permute(0, 3, 4, 1, 2).contiguous()


class PixelTransformer(nn.Module):
    def __init__(self, inputs, hidden=32, outputs=2):
        super().__init__()
        if hidden % 2:
            raise ValueError('Transformer hidden size must be even')
        self.hidden = hidden
        self.embed = nn.Linear(inputs, hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 2, hidden * 2, dropout=0.1, batch_first=True) for _ in range(2)])
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        b, t, c, h, w = x.shape
        sequence = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        position = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.hidden, 2, device=x.device, dtype=x.dtype) * (-math.log(10000) / self.hidden))
        encoding = x.new_zeros(t, self.hidden)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        result = []
        for chunk in sequence.split(4096):
            value = self.embed(chunk) + encoding
            for layer in self.layers:
                value = layer(value, src_mask=mask)
            result.append(self.head(value))
        return torch.cat(result).reshape(b, h, w, t, -1).permute(0, 3, 4, 1, 2).contiguous()


def build(name, inputs, hidden=32):
    constructors = {'convlstm': ConvLSTM, 'lstm': PixelLSTM, 'transformer': PixelTransformer}
    if name not in constructors:
        raise ValueError('Unknown model')
    return constructors[name](inputs, hidden)


def loss(prediction, target, valid):
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    if not torch.all(finite[valid]):
        raise ValueError('Nonfinite value in a supervised prediction')
    terms = []
    for channel in range(2):
        active = valid[:, channel]
        if active.any():
            terms.append((prediction[:, channel][active] - target[:, channel][active]).square().mean())
    if not terms:
        raise ValueError('The batch has no valid targets')
    return torch.stack(terms).mean()


def validate(model, loader, device):
    totals = np.zeros((2, 2), dtype='float64')
    model.eval()
    with torch.inference_mode():
        for x, target, valid in loader:
            prediction = model(x.to(device))[:, -1].cpu()
            for channel in range(2):
                selected = valid[:, channel]
                error = prediction[:, channel][selected] - target[:, channel][selected]
                totals[channel] += [error.square().double().sum().item(), error.numel()]
    present = totals[:, 1] > 0
    if not present.any():
        raise ValueError('Validation has no observations')
    result = float((totals[present, 0] / totals[present, 1]).mean())
    if not np.isfinite(result):
        raise ValueError('Nonfinite validation loss')
    return result


def evaluate(model, loader, device, mean, deviation):
    sums = np.zeros((2, 6), dtype='float64')
    model.eval()
    with torch.inference_mode():
        for x, target, valid in loader:
            prediction = model(x.to(device))[:, -1].cpu().numpy()
            target, valid = target.numpy(), valid.numpy()
            for channel in range(2):
                truth = target[:, channel][valid[:, channel]].astype('float64') * deviation[channel] + mean[channel]
                estimated = prediction[:, channel][valid[:, channel]].astype('float64') * deviation[channel] + mean[channel]
                if not np.isfinite(estimated).all():
                    raise ValueError('Nonfinite predictions')
                error = estimated - truth
                sums[channel] += [len(truth), truth.sum(), (truth ** 2).sum(), (error ** 2).sum(), np.abs(error).sum(), 0]
    records = []
    for name, row in zip(['npp', 'uhi'], sums):
        n, total, squared, errors, absolute, _ = row
        variance = squared - total ** 2 / n if n else 0
        records.append({'target': name, 'cells': int(n), 'r2': 1 - errors / variance if variance > 1e-10 else np.nan, 'rmse': np.sqrt(errors / n) if n else np.nan, 'mae': absolute / n if n else np.nan})
    return records


def fit(name, x, y, cfg, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    years = np.asarray(cfg['years'])
    window = cfg.get('window', 3)
    patch = cfg.get('patch', 32)
    train_end, validation_end = cfg.get('train_end', 2017), cfg.get('validation_end', 2019)
    splits = split_years(years, train_end, validation_end, window)
    xm, xs = scale(x, np.flatnonzero(years <= train_end))
    ym, ys = scale(y, np.flatnonzero(years <= train_end))
    normalized_x, normalized_y = normalize(x, xm, xs), normalize(y, ym, ys)
    sets = {key: Tiles(normalized_x, normalized_y, ids, window, patch) for key, ids in splits.items()}
    loaders = {key: DataLoader(value, batch_size=args.batch, shuffle=key == 'train', num_workers=0) for key, value in sets.items()}
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build(name, x.shape[1], args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=True)
    best, best_state, history_rows = float('inf'), None, []
    for epoch in range(args.epochs):
        model.train()
        total, batches = 0.0, 0
        for inputs, target, valid in loaders['train']:
            inputs, target, valid = inputs.to(device), target.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(inputs)[:, -1], target, valid)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += value.item()
            batches += 1
        validation = validate(model, loaders['validation'], device)
        history_rows.append({'epoch': epoch + 1, 'train_batch_mean': total / batches, 'validation_mse': validation})
        if validation < best:
            best = validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step()
    if best_state is None:
        raise ValueError('No valid checkpoint was produced')
    model.load_state_dict(best_state)
    checkpoint = {'state': best_state, 'model': name, 'inputs': x.shape[1], 'hidden': args.hidden, 'window': window, 'patch': patch, 'features': list(cfg['features']), 'targets': ['npp', 'uhi'], 'x_mean': torch.tensor(xm), 'x_std': torch.tensor(xs), 'y_mean': torch.tensor(ym), 'y_std': torch.tensor(ys), 'train_end': train_end, 'validation_end': validation_end, 'seed': args.seed}
    torch.save(checkpoint, folder / 'model.pt')
    pd.DataFrame(history_rows).to_csv(folder / 'history.csv', index=False)
    records = []
    for label, loader in loaders.items():
        for row in evaluate(model, loader, device, ym, ys):
            records.append({'split': label, **row})
    pd.DataFrame(records).to_csv(folder / 'metrics.csv', index=False)
    pd.DataFrame([{'split': label, 'target_year': int(years[index])} for label, ids in splits.items() for index in ids]).to_csv(folder / 'years.csv', index=False)
    print(f'{name}: complete')


def train_main(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).name} train')
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', choices=['convlstm', 'lstm', 'transformer', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.hidden, args.threads) < 1 or args.lr <= 0:
        raise ValueError('Training settings must be positive')
    torch.set_num_threads(args.threads)
    cfg = config(args.config)
    x, y, _, _ = history(cfg)
    models = ['convlstm', 'lstm', 'transformer'] if args.model == 'all' else [args.model]
    for name in models:
        fit(name, x, y, cfg, args)


def annual_frames(anchors, requested, interpolate=False):
    years = np.array(sorted(anchors))
    result = []
    for year in requested:
        if year in anchors:
            result.append(anchors[year])
            continue
        if not interpolate:
            raise ValueError(f'Missing annual predictors for {year}. Supply annual inputs or explicitly enable interpolation.')
        right = int(np.searchsorted(years, year))
        if right == 0 or right == len(years):
            raise ValueError('Interpolation cannot extrapolate beyond the supplied years')
        lower, upper = int(years[right - 1]), int(years[right])
        weight = (year - lower) / (upper - lower)
        result.append(anchors[lower] * (1 - weight) + anchors[upper] * weight)
    return np.stack(result).astype('float32')


def tiled(model, values, mean, deviation, patch, device):
    _, _, height, width = values.shape
    output = np.full((2, height, width), np.nan, dtype='float32')
    model.eval()
    with torch.inference_mode():
        for top in range(0, height, patch):
            for left in range(0, width, patch):
                hh, ww = min(patch, height - top), min(patch, width - left)
                chunk = values[:, :, top:top + hh, left:left + ww]
                valid = np.isfinite(chunk).all(axis=(0, 1))
                if not valid.any():
                    continue
                block = np.zeros((1, values.shape[0], values.shape[1], patch, patch), dtype='float32')
                block[0, :, :, :hh, :ww] = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                value = model(torch.from_numpy(block).to(device))[0, -1, :, :hh, :ww].cpu().numpy()
                value = value * deviation[:, None, None] + mean[:, None, None]
                if not np.isfinite(value[:, valid]).all():
                    raise ValueError('Nonfinite predictions')
                output[:, top:top + hh, left:left + ww] = np.where(valid[None], value, np.nan)
    return output


def predict_main(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).name} predict')
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--years', type=int, nargs='+', required=True)
    parser.add_argument('--interpolate', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.scenario):
        raise ValueError('Invalid scenario identifier')
    torch.set_num_threads(args.threads)
    cfg = config(args.config)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if list(cfg['features']) != saved['features']:
        raise ValueError('Feature order differs from the checkpoint')
    x, _, mask, profile = history(cfg, targets=False)
    historical_years = list(cfg['years'])
    if min(args.years) <= historical_years[-1]:
        raise ValueError('Requested years must follow the historical period')
    anchors = {year: x[index] for index, year in enumerate(historical_years)}
    root = cfg['_base'] / cfg['future'] / args.scenario
    for year in cfg['future_years']:
        if year <= historical_years[-1]:
            raise ValueError('Future input years must follow history')
        fields = []
        for name, pattern in cfg['features'].items():
            if name in cfg.get('static', ['elevation', 'slope']):
                path = cfg['_base'] / cfg['history'] / pattern.format(year=historical_years[-1])
            else:
                path = root / str(year) / f'{name}.tif'
            value, _ = read(path, profile, cfg.get('nodata', [-9999]))
            value[~mask] = np.nan
            fields.append(value)
        anchors[int(year)] = np.stack(fields)
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build(saved['model'], saved['inputs'], saved['hidden']).to(device)
    model.load_state_dict(saved['state'])
    xm, xs = saved['x_mean'].numpy(), saved['x_std'].numpy()
    ym, ys = saved['y_mean'].numpy(), saved['y_std'].numpy()
    records = []
    for year in args.years:
        requested = list(range(year - saved['window'] + 1, year + 1))
        filled = [y for y in requested if y not in anchors]
        sequence = annual_frames(anchors, requested, args.interpolate)
        sequence = (sequence - xm[None, :, None, None]) / xs[None, :, None, None]
        predictions = tiled(model, sequence, ym, ys, saved['patch'], device)
        for channel, target in enumerate(saved['targets']):
            write(args.output / f'{args.scenario}_{target}_{year}.tif', predictions[channel], profile)
        records.append({'scenario': args.scenario, 'year': year, 'window': saved['window'], 'temporal_fill': 'linear' if filled else 'none', 'filled_years': ' '.join(map(str, filled))})
    pd.DataFrame(records).to_csv(args.output / 'inputs.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = {'train': train_main, 'predict': predict_main}
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in commands:
        subparsers.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    commands[args.command](remaining)


if __name__ == '__main__':
    main()
