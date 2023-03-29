# FitCam

FitCam is now using poetry :) 
Please follow the following instructions to use

## Using 
```
pip install
--index-url https://test.pypi.org/simple/
--extra-index-url https://pypi.org/simple
FitCam
```

## Running webcam 
requires data in data/

```
webcam -o output_file
```
will run the yoga app.

### Adding others positions

```
preprocess -a
```
will preprocess data in yoga_images_in and add the new pose in data/


## Developpement Poetry
Using pyenv 
```
pyenv install 3.9.10
pyenv local 3.9.10
```
or using conda
```
conda create --name poetry_env python=3.9
conda activate poetry_env
```

install dependancies : 
```
poetry install
```

test environnement
```
poetry run webcam
poetry run python fitcam/run_webcam.py
```
