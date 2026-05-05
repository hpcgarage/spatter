from ramble.appkit import *


class Spatter(ExecutableApplication):
    name = 'spatter'
    maintainers('plavin', 'jyoung3131')
    

    # Load from spack
    with when("package_manager_family=spack"):
        software_spec("spatter_openmp", pkg_spec="spatter@main backend=openmp")

    required_package('spatter')

    input_file(
        "cpu_stream_patterns",
        url='https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/main/standard-suite/basic-tests/cpu-stream.json',
        description="CPU Patterns for stream",
        expand=False
    )

    input_file(
        "cpu_uniform_patterns",
        url='https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/spatter-devel/standard-suite/basic-tests/cpu-ustride.json',
        description="CPU Patterns for Uniform Test",
        expand=False
    )

    input_file(
        "gpu_stream_patterns",
        url='https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/main/standard-suite/basic-tests/gpu-stream.json',
        description="GPU Patterns for stream",
        expand=False
    )

    input_file(
        "gpu_uniform_patterns",
        url='https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/spatter-devel/standard-suite/basic-tests/gpu-ustride.json',
        description="GPU Patterns for Uniform Test",
        expand=False
    )

    # Executable
    executable('spatter_openmp', 
                'spatter {a} -b openmp -f {f}  -r {r}  -t {t} -l {l} -s {s} -v {v} -w {w} -x {x} -y {y} -z {z} {args}', 
                use_mpi=False)

    # Workloads
    workload("spatter", executable="spatter_openmp", inputs=["cpu_stream_patterns", "cpu_uniform_patterns"])


    workload_variable('a', default='-a', description='Aggregate (default on)', workloads=['spatter'])
    workload_variable('b', default='serial', description='Backend', workloads=['spatter'], values = ['serial', 'openmp', 'cuda'])
    workload_variable('d', default='8', description='Delta', workloads=['spatter'])
    workload_variable('k', default='gather', description='Kernel', workloads=['spatter'], values = ['gather', 'scatter', 'gs', 'multigather', 'multiscatter'])
    workload_variable('l', default='1024', description='Set Number of Gathers or Scatters to Perform', workloads=['spatter'])
    workload_variable('r', default='10', description='Number of Runs', workloads=['spatter'])
    workload_variable('s', default='random', description='Set Random Seed', workloads=['spatter'])
    workload_variable('t', default='1', description='Number of Threads', workloads=['spatter'])
    workload_variable('v', default='1', description='Set Verbosity Level', workloads=['spatter'])
    workload_variable('w', default='1', description='Set Wrap', workloads=['spatter'])
    workload_variable('x', default='8', description='Delta Gather', workloads=['spatter'])
    workload_variable('y', default='8', description='Delta Scatter', workloads=['spatter'])
    workload_variable('z', default='1024', description='Local Work Size', workloads=['spatter'])
    workload_variable('args', default='', description='Additional arguments', workloads=['spatter'])

    # Context for summary output
    figure_of_merit_context('config_row',
                            regex=r'^\s*(?P<config_idx>\d+)\s+(?P<bytes>\d+)',
                            output_format='Config {config_idx}')

    # Categorical FOMs
    figure_of_merit('Backend',
                    log_file='{log_file}',
                    fom_regex=r'Backend:\s+(?P<backend>.+)',
                    group_name='backend',
                    units='')

    figure_of_merit('Compiler',
                    log_file='{log_file}',
                    fom_regex=r'Compiler:\s+(?P<compiler>.+)',
                    group_name='compiler',
                    units='')

    
    figure_of_merit('Bytes',
                    log_file='{log_file}',
                    fom_regex=r'^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+',
                    group_name='bytes',
                    units='bytes',
                    contexts=['config_row'])

    figure_of_merit('Time',
                    log_file='{log_file}',
                    fom_regex=r'^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+(?P<time>[0-9.eE+\-]+)\s+[0-9.eE+\-]+',
                    group_name='time',
                    units='s',
                    contexts=['config_row'])

    figure_of_merit('Bandwidth',
                    log_file='{log_file}',
                    fom_regex=r'^\s*(?P<config>\d+)\s+(?P<bytes>\d+)\s+[0-9.eE+\-]+\s+(?P<bw>[0-9.eE+\-]+)',
                    group_name='bw',
                    units='MB/s',
                    contexts=['config_row'])
    
    # Aggregate Summary FOM
    figure_of_merit('Mean Bandwidth',
                    log_file='{log_file}',
                    fom_regex=r'Aggregate Summary - Mean Bandwidth:\s+(?P<summary_bw>[0-9.eE+\-]+)',
                    group_name='summary_bw',
                    units='MB/s')

    # Success Criteria
    success_criteria('mean_bytes',
                        mode='fom_comparison',
                        fom_name='Mean Bandwidth',
                        formula='{value} > 0.0')
