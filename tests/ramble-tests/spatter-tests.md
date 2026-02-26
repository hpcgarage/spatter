# Ramble Spatter Tests

This document outlines how to create a Ramble spatter application and to perform a basic uniform stride and stream test on a CPU. Install Ramble (including python requirements) via https://ramble.readthedocs.io/en/latest/getting_started.html. A shell script is available to automate and run this test.

## Create Ramble Application
```
$ mkdir spatter_tests2
$ cd spatter_tests2

$ ramble repo create spatter
$ ramble repo add spatter

$ mkdir -p spatter/applications/spatter
$ curl -O spatter/applications/spatter/application.py https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/main/tests/ramble-tests/application.py
```


## Create Experiment Workspace

```
$ ramble workspace create -d tests -a

```

Download test files
```
$ mkdir tests/basic_tests

$ curl -o tests/basic_tests/cpu-ustride.json https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/main/standard-suite/basic-tests/cpu-ustride.json

$curl -o tests/basic_tests/cpu-stream.json https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/main/standard-suite/basic-tests/cpu-stream.json
```

## Define Experiments
```
$ ramble workspace manage experiments spatter --overwrite -e UniformStride -v f=$PWD/tests/basic_tests/cpu-ustride.json -v a

$ ramble workspace manage experiments spatter --overwrite -e Stream -v f=$PWD/tests/basic_tests/cpu-stream.json

$ ramble workspace setup
$ ramble workspace concretize
```

Output is generated in `experiments/spatter/spatter/{experiment name}/{experiment name}.out`