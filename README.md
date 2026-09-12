# heat-carbon
Heat-Carbon is Python software for urban heat island and vegetation carbon-sink analysis, scenario simulation, and decision support.

Prepared for the 12th China Graduate Contest on Smart-city Technology and Creative Design.

This software system supports annual vegetation productivity estimation, raster trend analysis, bivariate spatial analysis, tree-based model interpretation, and joint NPP/UHI simulation.

Python 3.10 or newer is required. 

Input rasters must already share their CRS, affine transform, and dimensions. No automatic resampling is performed. Declared raster masks and NoData values are respected. A value of zero is not treated as missing unless the input file declares it as NoData.

casa.py estimates annual vegetation productivity using CASA.

spatial.py performs raster processing, UHI calculation, trend analysis, and global or local bivariate Moran analysis.

trees.py prepares raster samples, compares tree-based models, and runs SHAP analysis for UHI and vegetation productivity.

deep.py contains ConvLSTM, LSTM, and Transformer models for training, evaluation, and scenario prediction.
