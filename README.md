

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
pyhon3 -m black src/

```
