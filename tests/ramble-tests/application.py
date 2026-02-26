from ramble.appkit import *


class Spatter(ExecutableApplication):
    name = 'spatter'
    maintainers('plavin', 'jyoung3131')
    # license_path = ''

    # Load from spack
    with when("package_manager_family=spack"):
        software_spec("spatter_openmp", pkg_spec="spatter@2.1 backend=openmp")

    # Executable
    executable('spatter_openmp', 'spatter -f {f}', use_mpi=False)


    # Workloads
    workload("spatter", executable="spatter_openmp")


    workload_variable('a', default='', description='Aggregate (default off)', workloads=['spatter'])
    workload_variable('b', default='serial', description='Backend', workloads=['spatter'], values = ['serial', 'openmp', 'cuda'])
    workload_variable('c', default='', description='Enable compression of pattern indices', workloads=['spatter'])
    workload_variable('d', default='8', description='Delta', workloads=['spatter'])
    workload_variable('e', default='', description='Set Boundary (limits max value of pattern using modulo)', workloads=['spatter'])
    workload_variable('f', default='', description='Pattern Input File', workloads=['spatter'])
    workload_variable('g', default='', description='Set Inner Gather Pattern', workloads=['spatter'])
    workload_variable('j', default='', description='Set Pattern Size (truncates pattern to pattern-size)', workloads=['spatter'])
    workload_variable('k', default='gather', description='Kernel', workloads=['spatter'], values = ['gather', 'scatter', 'gs', 'multigather', 'multiscatter'])
    workload_variable('l', default='1024', description='Set Number of Gathers or Scatters to Perform', workloads=['spatter'])
    workload_variable('m', default='', description='Set Amount of Dummy Shared Memory to Allocate on GPUs', workloads=['spatter'])
    workload_variable('n', default='', description='Specify the Configuration Name', workloads=['spatter'])
    workload_variable('p', default='', description='Set Pattern', workloads=['spatter'])
    workload_variable('r', default='10', description='Number of Runs', workloads=['spatter'])
    workload_variable('s', default='random', description='Set Random Seed', workloads=['spatter'])
    workload_variable('t', default='1', description='Number of Threads', workloads=['spatter'])
    workload_variable('u', default='', description='Set Inner Scatter Pattern', workloads=['spatter'])
    workload_variable('v', default='1', description='Set Verbosity Level', workloads=['spatter'])
    workload_variable('w', default='1', description='Set Wrap', workloads=['spatter'])
    workload_variable('x', default='8', description='Delta Gather', workloads=['spatter'])
    workload_variable('y', default='8', description='Delta Scatter', workloads=['spatter'])
    workload_variable('z', default='1024', description='Local Work Size', workloads=['spatter'])

    # Figures -- Need improvement

# figure_of_merit_context(
#         "size_context",
#         regex=r"^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+(?P<time_s>[0-9.eE+-]+)\s+(?P<bw_mb_s>[0-9.eE+-]+)",
#         output_format="{bytes} bytes"
#     )

# figure_of_merit(
#         "Config",
#         log_file="{experiment_run_dir}/{experiment_name}.out",
#         fom_regex=r"^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+(?P<time_s>[0-9.eE+-]+)\s+(?P<bw_mb_s>[0-9.eE+-]+)",
#         group_name="config",
#         contexts=["size_context"]
#     )

# figure_of_merit(
#         "Bytes",
#         log_file="{experiment_run_dir}/{experiment_name}.out",
#         fom_regex=r"^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+(?P<time_s>[0-9.eE+-]+)\s+(?P<bw_mb_s>[0-9.eE+-]+)",
#         group_name="bytes",
#         contexts=["size_context"]
#     )

# figure_of_merit(
#         "Time",
#         log_file="{experiment_run_dir}/{experiment_name}.out",
#         fom_regex=r"^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+(?P<time_s>[0-9.eE+-]+)\s+(?P<bw_mb_s>[0-9.eE+-]+)",
#         group_name="s",
#         contexts=["size_context"]
#     )

# figure_of_merit(
#         "Bandwidth",
#         log_file="{experiment_run_dir}/{experiment_name}.out",
#         fom_regex=r"^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+(?P<time_s>[0-9.eE+-]+)\s+(?P<bw_mb_s>[0-9.eE+-]+)",
#         group_name="bw_mb_s",
#         units="MB/s",
#         contexts=["size_context"]
#     )