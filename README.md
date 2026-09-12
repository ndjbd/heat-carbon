# heat-carbon
Heat-Carbon is Python software for urban heat island and vegetation carbon-sink analysis, scenario simulation, and decision support.

Prepared for the 12th China Graduate Contest on Smart-city Technology and Creative Design.

Small programs for annual productivity estimation, raster trends, bivariate spatial analysis, tree-model interpretation, and joint NPP/UHI simulation.

The source files contain no comments or docstrings. Instructions are kept here. No private observations, trained research weights, personal paths, or original vegetation calibration table are included. This implementation changes preprocessing and evaluation choices. It does not reproduce previously reported paper metrics without a new run on the corresponding data.

Setup

Python 3.10 or newer is required. The tested environment is recorded in tested.json.

python -m pip install -r requirements.txt

optional.txt adds XGBoost, CatBoost, and pytest. They are only needed for the corresponding optional model comparisons or tests. Install a PyTorch build appropriate for your CPU or GPU. The delivered programs were exercised on CPU, not CUDA.

Run commands from this directory. All paths in configuration and manifests are resolved relative to those files. Input rasters must already share their CRS, affine transform, and dimensions. No automatic resampling is performed. Declared raster masks and NoData values are respected. A value of zero is not treated as missing unless the input file declares it as NoData.

casa.py estimates annual vegetation productivity using CASA.

spatial.py performs raster processing, UHI calculation, trend analysis, and global or local bivariate Moran analysis.

trees.py prepares raster samples, compares tree-based models, and runs SHAP analysis for UHI and vegetation productivity.

deep.py contains ConvLSTM, LSTM, and Transformer models for training, evaluation, and scenario prediction.
