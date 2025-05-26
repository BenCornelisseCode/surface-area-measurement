# surface-area-measurement

A toolkit for measuring surface area from images and video, built on top of Facebook’s [SAM 2.0](https://github.com/facebookresearch/sam2).

## 🚀 Features

- Batch‐process videos with SAM2 video predictor  
- Calibration routines for single‐camera setups  
- Contour and watershed‐based segmentation pipelines  
- Live‐stream inference with optional motor control  

## 📁 Project Structure

surface-area-measurement/
├── src/
│ └── surface_area_measurement/
│ ├── init.py
│ ├── validationSam2Testing.py
│ └── … your modules …
├── sam2/ # git submodule for facebookresearch/sam2
├── submodules/your-helper/ # any other helpers
├── setup.py # or pyproject.toml
└── README.md


## ⚙️ Installation

```bash
# clone with submodules
git clone --recurse-submodules https://github.com/BenCornelisseCode/surface-area-measurement.git
cd surface-area-measurement

# create & activate a venv
python3 -m venv .venv
source .venv/bin/activate

# install your package + SAM2 + helpers in editable mode
pip install --upgrade pip
pip install -e .
pip install -e sam2
pip install -e submodules/your-helper  # if you have extra submodules

# (optional) install dev tools:
pip install -e ".[dev]"
