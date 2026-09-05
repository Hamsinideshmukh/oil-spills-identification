# SAR Oil Spills Identification

A classical computer vision pipeline that detects oil spills in SAR (Synthetic Aperture Radar) satellite imagery — no deep learning, fully interpretable, built from hand-engineered filters, thresholding, morphology, and shape/texture descriptors.

**Tech stack:** Python · OpenCV · NumPy · SciPy

## The problem

Oil dampens the sea's capillary waves, so spills show up as dark patches in SAR imagery — but so do natural look-alikes (low-wind zones, algae blooms, biogenic slicks). This pipeline tells them apart using **shape** (real spills get smeared into elongated forms by wind/currents) and **texture** (spills are unnaturally smooth), not just brightness.

## Pipeline

1. **Preprocess** — log transform (turns multiplicative speckle noise into additive), despeckling via a Lee filter or a vectorized adaptive median filter, gamma correction to boost contrast in dark tones
2. **Segment** — Otsu or local adaptive thresholding, followed by morphological open/close to clean the mask
3. **Describe** — connected-component labeling, then per-blob features: area, circularity, a statistical smoothness (texture) score, mean intensity
4. **Classify** — minimum-distance classifier over (smoothness, circularity, log-area) against fitted class prototypes

## Structure

```
pipeline.py            # CLI — runs the full pipeline, fits prototypes, computes IoU
src/
├── filters.py          # despeckling + contrast transforms
├── segment.py           # thresholding
├── morphology.py        # mask cleanup
├── descriptors.py       # connected components + feature extraction
└── classify.py          # prototype fitting + classifier
```

## Usage

```bash
pip install -r requirements.txt

python pipeline.py --fit              # fit prototypes from labeled Class_0/Class_1 data
python pipeline.py --image chip.png   # run on one chip, saves debug masks + JSON summary
python pipeline.py                    # batch-score a sample folder
```

## Highlights

- Two despeckling implementations: a reference adaptive median filter and a vectorized version built for sub-500ms latency per chip
- Every step is physically motivated (SAR noise model, wave-damping texture), not a black box
- Modular phases — classifier is swappable for a learned model (SVM/CNN) without touching preprocessing or segmentation

#### Built with old-school image processing!
