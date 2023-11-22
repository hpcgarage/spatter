#include "Spatter/Benchmark.hh"
#include "Spatter/Input.hh"
#include "Spatter/Report.hh"

int main(int argc, char **argv) {

  const unsigned long warmup_runs = 10;
  bool timed = 0;

  Spatter::ClArgs cl;
  if (Spatter::parse_input(argc, argv, cl) != 0)
    return -1;

  for (std::unique_ptr<Spatter::ConfigurationBase> const &config : cl.configs) {
    std::cout << *config << std::endl;

    config->setup();
    for (unsigned long run = 0; run < (config->nruns + warmup_runs); ++run) {

      if (run >= warmup_runs)
        timed = 1;
      else
        timed = 0;

      if (config->run(timed) != 0)
        return -1;
    }
    config->report();
  }
}
