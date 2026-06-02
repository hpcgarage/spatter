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

#### Spack Download of Spatter (Optional)

> [!NOTE] Ramble will download Spatter once you run `ramble workspace concretize`. However, if you want to install spack elsewhere, you can now use spack to install the latest Spack-supported binary!

```
spack install spatter
==> Installing "clingo-bootstrap@=spack~apps~docs+ipo+optimized+python+static_libstdcpp build_system=cmake build_type=Release commit=2a025667090d71b2c9dce60fe924feb6bde8f667 generator=make patches:=bebb819,ec99431 platform=linux os=centos7 target=aarch64" from a buildcache
.....
[+] yukanvb spatter@main /home/vscode/spack/opt/spack/linux-aarch64/spatter-main-yukanvbkorcmawgzraw3bgtwnscmb22q (6s)
````

## Ramble Installation

```
git clone -c feature.manyFiles=true --depth=2 https://github.com/GoogleCloudPlatform/ramble.git ~/ramble
pip install -r ~/ramble/requirements.txt
# Source the path for the ramble executable
. ~/ramble/share/ramble/setup-env.sh
ramble -V
0.6.0 (625530ee2111373a7a4d121551051b1cd4181583)
```

### Ramble Local Test

The python file `example_ramble_test.py` automatically creates a workspace and runs both experiments. You can run it with the following commands:
```
cd $SPATTER_GIT/
# Create a temporary or work directory to hold your ramble repo and test results
mkdir tmp && cd tmp/
# Run the example Python script, which creates a ramble workspace and runs experiments
python3 ../tests/ramble/example_ramble_test.py
```


## Creating the Ramble Application:

Run the following commands to setup a new Ramble repo and copy over the application.py. Application.py contains pointers to the Spack package for Spatter, input files, and figures of merit to measure. 

```
cd $SPATTER_GIT/
# Create a temporary or work directory to hold your ramble repo
mkdir tmp && cd tmp/
ramble repo create r_spatter
==> Created applications and modifiers repo with namespace 'r_spatter'.
==> To register it with ramble, run this command:
  ramble repo add /path/to/tmp/r_spatter
```

```
ramble repo add r_spatter
==> Added applications repo with namespace 'r_spatter'.
...
==> Added base_platforms repo with namespace 'r_spatter'.
```

```
mkdir -p r_spatter/applications/spatter
cp $SPATTER_GIT/tests/ramble/application.py r_spatter/applications/spatter/.
```

## Create workspace and experiments:
Run the following commands:
```
$ ramble workspace create -d tests -a
==> Created and activated workspace in /workspaces/spatter/tmp/tests
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
==> Streaming details to log:
==>   /workspaces/spatter/tmp/tests/logs/setup.2026-06-02_01.02.38.out
==>   Setting up 2 out of 2 experiments:
==> Experiment #1 (1/2):
==>     name: spatter.spatter.UniformStride
==>     root experiment_index: 1
==>     log file: /workspaces/spatter/tmp/tests/logs/setup.2026-06-02_01.02.38/spatter.spatter.UniformStride.out
Experiment complete: 100%|=================================================================================================| Elapsed (s): 2.88
==>   Returning to log file: /workspaces/spatter/tmp/tests/logs/setup.2026-06-02_01.02.38.out
==> Experiment #2 (2/2):
==>     name: spatter.spatter.Stream
==>     root experiment_index: 2
==>     log file: /workspaces/spatter/tmp/tests/logs/setup.2026-06-02_01.02.38/spatter.spatter.Stream.out
Experiment complete: 100%|=================================================================================================| Elapsed (s): 0.27
==>   Returning to log file: /workspaces/spatter/tmp/tests/logs/setup.2026-06-02_01.02.38.out


```
$ ramble workspace concretize
```

Run the ramble experiments:
```
$ ramble on
```

Analyze the results and report the Figures of Merit

```
 ramble workspace analyze -f json
==> Streaming details to log:
==>   /workspaces/spatter/tmp/tests/logs/analyze.2026-06-02_01.09.09.out
==>   Analyzing 2 out of 2 experiments:
==> Experiment #1 (1/2):
==>     name: spatter.spatter.UniformStride
==>     root experiment_index: 1
==>     log file: /workspaces/spatter/tmp/tests/logs/analyze.2026-06-02_01.09.09/spatter.spatter.UniformStride.out
Processing phase analyze_experiments (1/6):  17%|============                                                              | Elapsed (s): 0.00==> Reading experiment results from cache file
...
==> Symlinks updated:
==>   /workspaces/spatter/tmp/tests/results/results.latest.json
```

```
ramble results report --fom
```