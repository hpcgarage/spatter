from benchpark.directives import variant, maintainers
from benchpark.experiment import Experiment
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType

class Spatter(
    Experiment,
    ProgrammingModel(
        ProgrammingModelType.Mpionly,
        ),
    ):


    variant(
        "pattern",
        default="stream",
        values=('stream', 'uniform'),
        description="Which ramble workload to execute"
    )

    variant(
        "backend",
        default="openmp",
        values = ("openmp", "cuda", "serial"),
        description="Configuration String."
    )
    variant(
        "cuda_arch",
        default="none",
        values=lambda x: True, # Allow custom input
        description="Which cuda architecture to use"
    )
    variant(
        "version",
        default="main",
        values = ("main", "devel"),
        description="Which Version to use."
    )

    maintainers("plavin", "jyoung3131")


    def compute_applications_section(self):

        
        # Define pattern file
        if self.spec.satisfies("pattern=stream"):
            if self.spec.satisfies("backend=cuda"):
                self.add_experiment_variable("f", "{input_path}/gpu-stream.json")
            else:
                self.add_experiment_variable("f", "{input_path}/cpu-stream.json")
        elif self.spec.satisfies("pattern=uniform"):
            if self.spec.satisfies("backend=cuda"):
                self.add_experiment_variable("f", "{input_path}/gpu-ustride.json")
            else:
                self.add_experiment_variable("f", "{input_path}/cpu-ustride.json")

        # n_gpus
        if self.spec.satisfies("backend=cuda"):
            self.add_experiment_variable("n_gpus", "{n_resources}", True)  
        else: 
            self.add_experiment_variable("n_gpus", "0") 


        self.add_experiment_variable("n_nodes", "1")
        self.add_experiment_variable("n_ranks", "1")
        self.add_experiment_variable("processes_per_node", "1")



        # Satisfy the base class required variables
        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="4194304",
            total_problem_size="4194304"
        )

        # # 2. Write the file to the current working directory.
        # # When 'benchpark setup' runs, it executes this Python script 
        # # from the root of the experiment's staging area.

    def compute_package_section(self):

        version = self.spec.variants['version'][0]
        
        # Backend
        if self.spec.satisfies("backend=openmp"):
            backend = "backend=openmp"
        elif self.spec.satisfies("backend=cuda") and not self.spec.satisfies("cuda_arch=none"):
            backend = f"backend=cuda cuda_arch={self.spec.variants['cuda_arch'][0]}"
        else:
            backend = "backend=serial"

        if self.spec.satisfies("+mpi"):
            mpi_flag = '+mpi'
        else:
            mpi_flag = '~mpi'

        self.add_package_spec(
            self.name,
            [
                f"spatter@{version} {backend} {mpi_flag}"
            ]
        )