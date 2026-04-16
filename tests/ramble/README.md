# Testing with Ramble

## Prereqs:
- (Recommended): Create and activate a python environment
- Install Ramble following the instructions in https://ramble.readthedocs.io/en/latest/getting_started.html#installation
- Install Spack following the instructions in https://spack.io/about/#install-spack


## Creating the Ramble Application:
Run the following commands and then copy `application.py` into `spatter/applications/spatter`
```
$ ramble repo create spatter
$ ramble repo add spatter

$ mkdir -p spatter/applications/spatter
```

## Create workspace and experiments:
Run the following commands:
```
$ ramble workspace create -d tests -a
$ ramble workspace manage experiments spatter --overwrite -e UniformStride -v f=$PWD/tests/input/cpu-ustride.json 
$ ramble workspace manage experiments spatter --overwrite -e Stream -v f=$PWD/tests/input/cpu-stream.json

//Run experiment
$ ramble workspace setup
$ ramble workspace concretize
$ ramble on

```

The python file `example_ramble_test.py` automatically creates a workspace and runs both experiment.