# FitCam

FitCam is now using poetry :) 
Please follow the following instructions to use

## Poetry
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
