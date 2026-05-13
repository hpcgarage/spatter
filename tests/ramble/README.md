# Testing with Ramble

## High-level steps
- Create and activate a python environment
- Install Ramble following the instructions in https://ramble.readthedocs.io/en/latest/getting_started.html#installation
- Install Spack following the instructions in https://spack.io/about/#install-spack

SPATTER_GIT - refers to the top-level Git repo for Spatter

## Creating the Ramble Application:
Run the following commands to setup a new experiment for spatter. `application.py` into `spatter/applications/spatter`
```
SPATTER_GIT$ ramble repo create spatter
SPATTER_GIT$ ramble repo add spatter

SPATTER_GIT$ mkdir -p spatter/applications/spatter

SPATTER_GIT$ cp tests/ramble/application.py spatter/applications/spatter/.
```

## Create workspace and experiments:
Run the following commands:
```
$ ramble workspace create -d tests -a
==> Created and activated workspace in /workspaces/spatter/tests
$ ramble workspace manage experiments spatter --overwrite -e UniformStride -v f=$PWD/tests/inputs/cpu-ustride.json 
$ ramble workspace manage experiments spatter --overwrite -e Stream -v f=$PWD/tests/inputs/cpu-stream.json
```

```
$ ramble config add "variants:package_manager:spack"
$ ramble config add "software:packages:spatter:pkg_spec:'spatter@develop backend=openmp'"
$ ramble config add "software:environments:spatter:packages:[spatter]"
```

```
//Run experiment
$ ramble workspace setup
$ ramble workspace concretize
$ ramble on

```

The python file `example_ramble_test.py` automatically creates a workspace and runs both experiments.

## Spack Installation details

### Prerequisites

If you are building Spatter in a container, you will need to following base dependencies, along with a compiler (GCC).
```
apt install python3.12 python3.12-dev python3.12-venv python3-pip cmake
```


### Spack Install

. spack/share/spack/setup-env.sh

```
spack external find python
...
==> The following specs have been detected on this system and added to /home/vscode/.spack/packages.yaml
-- no arch / no compilers ---------------------------------------
python@3.12.3

spack compiler find
==> Added 1 new compiler to /home/vscode/.spack/packages.yaml
    gcc@13.3.0
==> Compilers are defined in the following files:
    /home/vscode/.spack/packages.yaml

spack external find cmake
==> The following specs have been detected on this system and added to /home/vscode/.spack/packages.yaml
-- no arch / no compilers ---------------------------------------
cmake@3.28.3
```

```
$ spack env create spatter
==> Created environment spatter in: /workspaces/spatter/spack_tmp/spack/var/spack/environments/spatter
==> Activate with: spack env activate spatter
```

It also is likely useful to create a Python environment to isolate any pip installed packages
```
$ python3 -m venv pyenv_spatter
vscode ➜ /workspaces/spatter (benchpark) $ . pyenv_spatter/bin/activate
```

### Install Ramble

```
git clone -c feature.manyFiles=true https://github.com/GoogleCloudPlatform/ramble.git
pip install -r ramble/requirements.txt
# Source the path for the ramble executable
$ . ramble/share/ramble/setup-env.sh
$ ramble -V
0.6.0 (ff9672ec8267ef4de83387f26f4d4d62e11d824a)
```