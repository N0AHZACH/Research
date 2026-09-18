# `experiments`

Command-line entry points live here.

- `train.py`: train one model from one config.
- `evaluate.py`: evaluate models from a manifest.
- `analyze_routing.py`: inspect DLR routing behavior.
- `make_figures.py`: generate paper figures and tables from artifacts.

These files should stay thin. Shared logic belongs in `new/src/dlr/`.
