# Testing with Ramble
This README covers testing with Ramble independent of other tools (like BenchPark). Running Spatter with Benchpark will use similar application and experiment files but will have different steps.

## High-level steps
1) Install Spack, which can be used to download a release version of Spatter
2) Create and activate a python environment, which helps to create a clean test environment
3) Install Ramble, which can be used to run experiments
4) Test using the `example_ramble_test.py` script. 

## Key files/folders
- `SPATTER_GIT` - refers to the top-level Git repo for Spatter

## Prerequisites

If you are building Spatter in a container, you will need to following base dependencies, along with a compiler (GCC).
```
sudo apt install python3.12 python3.12-dev python3.12-venv python3-pip cmake
```

## Spack Installation

```
$ git clone --depth=2 https://github.com/spack/spack.git ~/spack
cd ~/spack
# Run the setup environment script to add to your path
. spack/share/spack/setup-env.sh
```

Then you want to let spack know which python and compilers you are using to compile and run code:

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

You can then save a collection of these package options for later use with spack:
```
$ spack env create spatter
==> Created environment spatter in: /workspaces/spatter/spack_tmp/spack/var/spack/environments/spatter
==> Activate with: spack env activate spatter
```

It also is likely useful to create a Python environment to isolate any pip installed packages. 
```
cd SPATTER_GIT
python3 -m venv pyenv_spatter
vscode ➜ /workspaces/spatter (benchpark) $ . pyenv_spatter/bin/activate
```

#### Spack Download of Spatter

```
spack install spatter
==> Installing "clingo-bootstrap@=spack~apps~docs+ipo+optimized+python+static_libstdcpp build_system=cmake build_type=Release commit=2a025667090d71b2c9dce60fe924feb6bde8f667 generator=make patches:=bebb819,ec99431 platform=linux os=centos7 target=aarch64" from a buildcache
.....
[+] yukanvb spatter@main /home/vscode/spack/opt/spack/linux-aarch64/spatter-main-yukanvbkorcmawgzraw3bgtwnscmb22q (6s)
````

### Ramble Installation

```
git clone -c feature.manyFiles=true --depth=2 https://github.com/GoogleCloudPlatform/ramble.git ~/ramble
pip install -r requirements.txt
# Source the path for the ramble executable
. ramble/share/ramble/setup-env.sh
$ ramble -V
0.6.0 (625530ee2111373a7a4d121551051b1cd4181583)
```

### Ramble Local Test

The python file `example_ramble_test.py` automatically creates a workspace and runs both experiments. You can run it with the following commands:
```

```


## Creating the Ramble Application:
Run the following commands to setup a new experiment for spatter. 
```
cd $SPATTER_GIT/
mkdir tmp
cd tmp/
ramble repo create spatter
$ ramble repo add spatter
mkdir -p spatter/applications/spatter
$ cp $SPATTER_GIT/tests/ramble/application.py spatter/applications/spatter/.
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