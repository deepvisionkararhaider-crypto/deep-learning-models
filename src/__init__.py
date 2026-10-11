"""Deep Learning, Medical AI & Smart Agriculture platform (source package).

Layout
------
src.registry   -> single source of truth for every model (task, dataset, IO contract)
src.artifacts  -> save/load trained weights + metadata, with graceful missing handling
src.imaging    -> shared image preprocessing (identical for training and inference)
src.datasets   -> real public dataset loaders
src.models     -> PyTorch architectures
src.ui         -> Streamlit page renderers
"""

__version__ = "2.0.0"
