import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.special import ndtr
from scipy.sparse import csr_matrix


def read(path, reference=None, nodata=()):
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        if reference is not None:
            aligned = (
                src.width == reference['width']
                and src.height == reference['height']
                and src.crs == reference['crs']
                and src.transform.almost_equals(reference['transform'], precision=1e-9)
            )
            if not aligned:
                raise ValueError('Raster grids do not match')
        if src.count != 1:
            raise ValueError('A single-band raster is required')
        value = src.read(1, masked=True).astype('float64').filled(np.nan)
    for item in nodata:
        value[value == item] = np.nan
    value[~np.isfinite(value)] = np.nan
    return value, profile


def write(path, value, reference, dtype='float32', nodata=-9999):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {key: reference[key] for key in ('width', 'height', 'crs', 'transform')}
    profile.update(driver='GTiff', count=1, dtype=dtype, nodata=nodata, compress='lzw')
    result = np.where(np.isfinite(value), value, nodata).astype(dtype)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(result, 1)


def standardize(value, mask=None):
    valid = np.isfinite(value)
    if mask is not None:
        valid &= mask
    if valid.sum() < 2:
        raise ValueError('At least two valid cells are required')
    deviation = value[valid].std()
    if not np.isfinite(deviation) or deviation <= 0:
        raise ValueError('The raster has zero variance')
    result = np.full(value.shape, np.nan, dtype='float64')
    result[valid] = (value[valid] - value[valid].mean()) / deviation
    return result


def config(path):
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        result = json.load(stream)
    result['_base'] = path.parent
    years = np.asarray(result['years'], dtype=int)
    if len(years) < 3 or not np.all(np.diff(years) == 1):
        raise ValueError('Historical years must be consecutive')
    if result.get('heat_kind', 'uhi') not in ('lst', 'uhi'):
        raise ValueError('heat_kind must be lst or uhi')
    return result


def history(cfg, targets=True):
    root = cfg['_base'] / cfg['history']
    features = cfg['features']
    years = np.asarray(cfg['years'], dtype=int)
    profile, inputs, outputs = None, [], []
    spatial_mask = None
    for year in years:
        frame = []
        for pattern in features.values():
            value, profile = read(root / pattern.format(year=year), profile, cfg.get('nodata', [-9999]))
            frame.append(value)
        if spatial_mask is None:
            spatial_mask = np.ones(frame[0].shape, dtype=bool)
            if cfg.get('mask'):
                mask, _ = read(cfg['_base'] / cfg['mask'], profile)
                spatial_mask = np.isfinite(mask) & (mask > 0)
        frame = np.stack(frame)
        frame[..., ~spatial_mask] = np.nan
        inputs.append(frame)
        if targets:
            npp, _ = read(root / cfg['npp'].format(year=year), profile, cfg.get('nodata', [-9999]))
            heat, _ = read(root / cfg['heat'].format(year=year), profile, cfg.get('nodata', [-9999]))
            if cfg.get('heat_kind', 'uhi') == 'lst':
                heat = standardize(heat, spatial_mask)
            target = np.stack([npp, heat])
            target[..., ~spatial_mask] = np.nan
            outputs.append(target)
    x = np.stack(inputs).astype('float32')
    y = np.stack(outputs).astype('float32') if targets else None
    return x, y, spatial_mask, profile


def calculate(values, years, minimum):
    values = np.asarray(values, dtype='float64')
    values = np.where(np.isfinite(values), values, np.nan)
    years = np.asarray(years, dtype='float64')
    if values.shape[0] != len(years) or not np.all(np.diff(years) > 0):
        raise ValueError('Years must be strictly increasing and match the stack')
    a, b = np.triu_indices(len(years), 1)
    differences = values[b] - values[a]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        slope = np.nanmedian(differences / (years[b] - years[a])[:, None], axis=0)
    score = np.nansum(np.sign(differences), axis=0)
    count = np.isfinite(values).sum(axis=0)
    ordered = np.sort(values, axis=0)
    correction = np.zeros(values.shape[1])
    run = np.zeros(values.shape[1], dtype='int64')
    for index in range(len(years)):
        finite = np.isfinite(ordered[index])
        same = finite & (ordered[index] == ordered[index - 1]) if index else np.zeros_like(finite)
        correction += np.where(~same, run * (run - 1) * (2 * run + 5), 0)
        run = np.where(finite, np.where(same, run + 1, 1), 0)
    correction += run * (run - 1) * (2 * run + 5)
    variance = (count * (count - 1) * (2 * count + 5) - correction) / 18.0
    z = np.zeros(values.shape[1])
    np.divide(score - np.sign(score), np.sqrt(np.maximum(variance, 0)), out=z, where=variance > 0)
    p = 2 * ndtr(-np.abs(z))
    category = np.zeros(values.shape[1], dtype='uint8')
    category[slope > 0] = 3
    category[slope < 0] = 4
    category[(slope > 0) & (p < 0.1)] = 2
    category[(slope < 0) & (p < 0.1)] = 5
    category[(slope > 0) & (p < 0.05)] = 1
    category[(slope < 0) & (p < 0.05)] = 6
    valid = (count >= minimum) & np.isfinite(slope)
    return np.where(valid, slope, np.nan), np.where(valid, z, np.nan), np.where(valid, p, np.nan), np.where(valid, category, 255).astype('uint8')


def trend_main(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).name} trend')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--years', type=int, nargs='+', required=True)
    parser.add_argument('--pattern', default='{year}.tif')
    parser.add_argument('--minimum', type=int)
    parser.add_argument('--block', type=int, default=64)
    args = parser.parse_args(argv)
    minimum = args.minimum if args.minimum is not None else len(args.years)
    if not 3 <= minimum <= len(args.years) or args.block < 1:
        raise ValueError('Invalid minimum count or block size')
    if not np.all(np.diff(args.years) > 0):
        raise ValueError('Years must be strictly increasing')
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(args.input / args.pattern.format(year=y))) for y in args.years]
        first = sources[0]
        for src in sources:
            if (src.count != 1 or src.shape != first.shape or src.crs != first.crs
                    or not src.transform.almost_equals(first.transform, precision=1e-9)):
                raise ValueError('Raster grids do not match')
        profile = dict(driver='GTiff', height=first.height, width=first.width, count=1, crs=first.crs, transform=first.transform, compress='lzw')
        outputs = [stack.enter_context(rasterio.open(args.output / f'{name}.tif', 'w', **profile, dtype='float32', nodata=-9999)) for name in ('slope', 'z', 'p')]
        categories = stack.enter_context(rasterio.open(args.output / 'class.tif', 'w', **profile, dtype='uint8', nodata=255))
        totals = np.zeros(7, dtype='int64')
        for top in range(0, first.height, args.block):
            for left in range(0, first.width, args.block):
                window = Window(left, top, min(args.block, first.width - left), min(args.block, first.height - top))
                shape = (int(window.height), int(window.width))
                values = np.stack([src.read(1, window=window, masked=True).astype('float64').filled(np.nan).ravel() for src in sources])
                values[(values == -9999) | ~np.isfinite(values)] = np.nan
                slope, z, p, codes = calculate(values, args.years, minimum)
                for dst, value in zip(outputs, (slope, z, p)):
                    dst.write(np.where(np.isfinite(value), value, -9999).reshape(shape).astype('float32'), 1, window=window)
                categories.write(codes.reshape(shape), 1, window=window)
                totals += np.bincount(codes[codes != 255], minlength=7)
    labels = ['zero_sen_slope', 'significant_increase', 'weak_increase', 'nonsignificant_increase', 'nonsignificant_decrease', 'weak_decrease', 'significant_decrease']
    summary = pd.DataFrame({'code': range(7), 'label': labels, 'cells': totals, 'percent': totals / max(int(totals.sum()), 1) * 100})
    summary.to_csv(args.output / 'summary.csv', index=False)


def weights(mask, rook=False):
    rows, cols = np.where(mask)
    index = np.full(mask.shape, -1, dtype='int64')
    index[rows, cols] = np.arange(len(rows))
    source, target = [], []
    offsets = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (not rook or abs(a) + abs(b) == 1)]
    for dy, dx in offsets:
        rr, cc = rows + dy, cols + dx
        inside = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
        chosen = np.flatnonzero(inside)
        neighbors = index[rr[inside], cc[inside]]
        linked = neighbors >= 0
        source.extend(chosen[linked])
        target.extend(neighbors[linked])
    matrix = csr_matrix((np.ones(len(source)), (source, target)), shape=(len(rows), len(rows)))
    keep = np.asarray(matrix.sum(axis=1)).ravel() > 0
    matrix = matrix[keep][:, keep].tocsr()
    degree = np.diff(matrix.indptr)
    matrix.data /= np.repeat(degree, degree)
    return matrix, rows[keep], cols[keep]


def statistics(x, y, matrix, permutations=999, seed=42):
    if len(x) < 3 or permutations < 1:
        raise ValueError('At least three connected cells and one permutation are required')
    if x.std() == 0 or y.std() == 0:
        raise ValueError('Both variables must have nonzero variance')
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    lag = matrix @ zy
    local = zx * lag
    observed = float(local.sum() / (len(x) - 1))
    generator = np.random.default_rng(seed)
    simulated = np.array([zx @ (matrix @ generator.permutation(zy)) / (len(x) - 1) for _ in range(permutations)])
    global_p = (1 + np.count_nonzero(np.abs(simulated) >= abs(observed) - 1e-12)) / (permutations + 1)
    local_p = np.ones(len(x))
    degrees = np.diff(matrix.indptr)
    for start in range(0, len(x), 256):
        stop = min(start + 256, len(x))
        ids = np.arange(start, stop)
        count = degrees[start:stop]
        picks = []
        sums = np.zeros((permutations, len(ids)))
        for step in range(int(count.max())):
            draw = generator.integers(0, len(x) - 1, size=sums.shape)
            duplicate = np.zeros(draw.shape, dtype=bool)
            for old in picks:
                duplicate |= draw == old
            while duplicate.any():
                draw[duplicate] = generator.integers(0, len(x) - 1, size=int(duplicate.sum()))
                duplicate[:] = False
                for old in picks:
                    duplicate |= draw == old
            picks.append(draw)
            chosen = draw + (draw >= ids[None, :])
            sums += zy[chosen] * (step < count)[None, :]
        sampled = zx[ids][None, :] * sums / count[None, :]
        expected = -zx[ids] * zy[ids] / (len(x) - 1)
        distance = np.abs(local[ids] - expected)
        exceed = (np.abs(sampled - expected) >= distance - 1e-12).sum(axis=0)
        local_p[ids] = (1 + exceed) / (permutations + 1)
    quadrants = np.select([(zx >= 0) & (lag >= 0), (zx < 0) & (lag >= 0), (zx < 0) & (lag < 0)], [1, 2, 3], default=4).astype('uint8')
    return observed, global_p, local, local_p, quadrants, zx, lag


def moran_main(argv=None):
    parser = argparse.ArgumentParser(prog=f'{Path(__file__).name} moran')
    parser.add_argument('heat', type=Path)
    parser.add_argument('npp', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--heat-kind', choices=['lst', 'uhi'], default='uhi')
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rook', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        raise ValueError('Alpha must be between zero and one')
    heat, profile = read(args.heat, nodata=(-9999,))
    npp, _ = read(args.npp, profile, (-9999,))
    if args.heat_kind == 'lst':
        heat = standardize(heat)
    valid = np.isfinite(heat) & np.isfinite(npp)
    matrix, rows, cols = weights(valid, args.rook)
    result = statistics(heat[rows, cols], npp[rows, cols], matrix, args.permutations, args.seed)
    value, p_global, local, p_local, quadrants, zx, lag = result
    codes = np.where(p_local < args.alpha, quadrants, 0).astype('uint8')
    for name, values in [('local_i', local), ('local_p', p_local), ('class', codes)]:
        output = np.full(heat.shape, np.nan)
        output[rows, cols] = values
        write(args.output / f'{name}.tif', output, profile, 'uint8' if name == 'class' else 'float32', 255 if name == 'class' else -9999)
    if args.heat_kind == 'lst':
        write(args.output / 'uhi.tif', heat, profile)
    pd.DataFrame([{'focal': 'uhi', 'neighbor': 'npp', 'cells': len(rows), 'excluded_islands': int(valid.sum() - len(rows)), 'i': value, 'p_two_sided': p_global, 'permutations': args.permutations}]).to_csv(args.output / 'global.csv', index=False)
    counts = np.bincount(codes, minlength=5)
    pd.DataFrame({'code': range(5), 'label': ['not_significant', 'HH', 'LH', 'LL', 'HL'], 'cells': counts, 'percent': 100 * counts / counts.sum()}).to_csv(args.output / 'summary.csv', index=False)
    pd.DataFrame({'row': rows, 'col': cols, 'z_uhi': zx, 'lag_z_npp': lag, 'local_i': local, 'p_two_sided': p_local, 'quadrant': quadrants, 'class': codes}).to_csv(args.output / 'cells.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = {'trend': trend_main, 'moran': moran_main}
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in commands:
        subparsers.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    commands[args.command](remaining)


if __name__ == '__main__':
    main()
