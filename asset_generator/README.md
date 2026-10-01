# Image to GLB tiles

Put photos in `input/`, then run:

```bash
pip install -r asset_generator/requirements.txt
python asset_generator/generate.py
```

The script reads every PNG, JPG, JPEG, WebP, BMP, and TIFF file directly inside
`input/`. It creates one textured `.glb` tile per photo in
`output/YYYYMMDD_HHMMSS_microseconds/`. Each photo is resized to fit the square
tile without cropping; unused space is white. The tile measures 11 × 11 × 0.6 cm.

The script only generates visual GLB assets. It does not run the simulator or a
model.

## SAPIEN preview

In an environment with the repository's ManiSkill/SAPIEN dependencies installed,
render every GLB from the most recent output batch with:

```bash
python asset_generator/preview.py
```

SAPIEN 3.0.0b1 also needs `pkg_resources`, so if this environment already has
setuptools 82 or newer, install `setuptools<82` in that environment first.

To render one file:

```bash
python asset_generator/preview.py asset_generator/output/<timestamp>/0001_photo.glb
```

Preview PNGs are saved in `output/<timestamp>/preview/`. This loads only each GLB
in a small SAPIEN scene and captures one camera frame; it does not run an
Act2Answer episode or load a VLA model.
