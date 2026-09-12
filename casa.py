import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spatial import read, write


FIELDS = ['ndvi_min', 'ndvi_max', 'sr_min', 'sr_max', 'efficiency']


def estimate(land, ndvi, radiation, temperature, precipitation, table, topt=None, floor=0.001):
    if not 0 <= floor < 0.95:
        raise ValueError('The FPAR floor must be between 0 and 0.95')
    if table['class'].duplicated().any() or not np.isfinite(table[['class'] + FIELDS]).all().all():
        raise ValueError('Invalid parameter table')
    if ((table.ndvi_max <= table.ndvi_min) | (table.sr_max <= table.sr_min)
            | (table.efficiency < 0)).any():
        raise ValueError('Invalid parameter ranges')
    parameters = np.full((5,) + land.shape, np.nan)
    for index, field in enumerate(FIELDS):
        for code, value in zip(table['class'], table[field]):
            parameters[index][land == code] = value
    lower, upper, sr_lower, sr_upper, efficiency = parameters
    optimum = temperature if topt is None else topt
    valid = np.isfinite(np.stack([land, ndvi, radiation, temperature, precipitation, optimum])).all(axis=0)
    valid &= np.isfinite(parameters).all(axis=0)
    valid &= (ndvi > -1) & (ndvi < 1) & (radiation >= 0) & (precipitation >= 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        sr = (1 + ndvi) / (1 - ndvi)
        fraction = 0.5 * (0.949 * (ndvi - lower) / (upper - lower) + 0.001
                          + 0.949 * (sr - sr_lower) / (sr_upper - sr_lower) + 0.001)
        fraction = np.clip(fraction, floor, 0.95)
        first = np.clip(0.8 + 0.02 * optimum - 0.0005 * optimum ** 2, 0, 1)
        first = np.where(temperature < -10, 0, first)
        second = 1.184 / (1 + np.exp(0.2 * (optimum - 10 - temperature)))
        second /= 1 + np.exp(0.3 * (temperature - optimum - 10))
        pet = radiation * 0.75 / 2.45
        water = np.divide(precipitation, pet, out=np.zeros_like(pet), where=pet > 0)
        result = radiation * 0.5 * fraction * efficiency * first * np.clip(second, 0, 1) * np.clip(water, 0, 1)
    return np.where(valid, result, np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('parameters', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fpar-min', type=float, default=0.001)
    args = parser.parse_args()
    rows = pd.read_csv(args.manifest)
    table = pd.read_csv(args.parameters)
    required = ['land', 'ndvi', 'radiation', 'temperature', 'precipitation']
    if not {'year', *required}.issubset(rows.columns) or not {'class', *FIELDS}.issubset(table.columns):
        raise ValueError('Missing columns in the manifest or parameter table')
    if rows.year.duplicated().any():
        raise ValueError('Years must be unique')
    results = []
    for row in rows.sort_values('year').to_dict('records'):
        arrays, profile = [], None
        for field in required:
            value, profile = read(args.manifest.parent / row[field], profile, (-9999,))
            arrays.append(value)
        optimum = None
        if pd.notna(row.get('topt')):
            optimum, _ = read(args.manifest.parent / row['topt'], profile, (-9999,))
        npp = estimate(*arrays, table, optimum, args.fpar_min)
        if not np.isfinite(npp).any():
            raise ValueError('No valid NPP values were produced')
        year = int(row['year'])
        write(args.output / f'npp_{year}.tif', npp, profile)
        results.append({'year': year, 'valid_cells': int(np.isfinite(npp).sum()), 'mean_npp': float(np.nanmean(npp))})
    pd.DataFrame(results).to_csv(args.output / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
