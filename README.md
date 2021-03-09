

## Requirements

- python3
- pyenv
- pip3

## Installing

```bash

# create the venv
python3 -m venv my-env

# activate env
source ./my-env/bin/activate

# deactivate env
deactivate

# install new packages
pip3 install some-package

# create a list of dependencies from installed packages
pip freeze > requirements.txt

# install deps from requirements
pip3 install -r requirements.txt


```

## Autoformatting


```bash
python3 -m black src/

```

## Data

Data is too large to put it in the repo, please download it and place it
in `train-data`


- `train-data/ravdess` -> https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio
- `train-data/cremad` -> https://www.kaggle.com/ejlok1/cremad


Original notebook https://www.kaggle.com/franleplant/notebookc90b5d379c

## Usage

This model requires the training features to be calculated, you can do that
by either using the `training_data_*.csv` previously generated, or generating new ones
by using `python3 src/get_training_data.py`.

After that is done (it is going to take a while) you need to train the model.
A pretrained model will be stored in `model.h5`, if you want to train it you 
can run `python3 src/model.py`.


Finally you can now evaluate new files by using `src/main.py`
