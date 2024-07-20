#include <iostream>
#include <string>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int omp_thread_test(int thread_count, int argc, char **argv) {
#ifndef USE_OPENMP
  (void)thread_count;
  (void)argc;
  (void)argv;
  return EXIT_SUCCESS;
#else
  Spatter::ClArgs cl;

  if (Spatter::parse_input(argc, argv, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1) {
    std::cerr << "Test failure on OMP Threads: Expected number of runs to "
                 "be "
              << 1 << ", actually was " << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr << "Test failure on OMP Threads: Failed to create or allocate "
                 "a ConfigurationBase object"
              << std::endl;
    return EXIT_FAILURE;
  }

  if (thread_count < 0 && cl.configs[0]->omp_threads < 0) {
    std::cerr << "Test failure on OMP Threads: user requested negative thread "
                 "count and request was granted."
              << std::endl;
    return EXIT_FAILURE;
  }

  if (thread_count == 0 &&
      cl.configs[0]->omp_threads != omp_get_max_threads()) {
    std::cerr
        << "Test failure on OMP Threads: requested 0 threads and should have "
           "defaulted to OMP Max Thread Count, but instead allocated "
        << cl.configs[0]->omp_threads << "threads." << std::endl;
    return EXIT_FAILURE;
  }

  if (thread_count > 0 && thread_count != 0 &&
      cl.configs[0]->omp_threads != thread_count &&
      thread_count < omp_get_max_threads()) {
    std::cerr << "Test failure on OMP Threads: requested " << thread_count
              << " threads but got " << cl.configs[0]->omp_threads
              << " threads instead." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
#endif
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int argc_ = 4;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  asprintf(&argv_[0], "./spatter");
  asprintf(&argv_[1], "-p1,2,3,4");
  asprintf(&argv_[2], "-bopenmp");

  optind = 1;
  asprintf(&argv_[3], "-t4");
  if (omp_thread_test(4, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  optind = 1;
  asprintf(&argv_[3], "-t0");
  if (omp_thread_test(0, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  optind = 1;
  asprintf(&argv_[3], "-t10000000");
  if (omp_thread_test(10000000, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  optind = 1;
  asprintf(&argv_[3], "-t-1");
  if (omp_thread_test(-1, argc_, argv_) == EXIT_SUCCESS)
    return EXIT_FAILURE;

  optind = 1;
  asprintf(&argv_[3], "--omp-threads=0");
  if (omp_thread_test(0, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  optind = 1;
  argc_ = 3;
  if (omp_thread_test(0, argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
