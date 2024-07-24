#include <iostream>
#include <string>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/Input.hh"

int parse_check(int argc_, char **argv_, Spatter::ClArgs &cl) {
  if (Spatter::parse_input(argc_, argv_, cl) != 0) {
    std::cerr << "Parse Input Failed" << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs.size() != 1) {
    std::cerr
        << "Test failure on Concurrent Pattern: Expected number of runs to "
           "be 1, actually was "
        << cl.configs.size() << std::endl;
    return EXIT_FAILURE;
  }

  if (cl.configs[0] == NULL) {
    std::cerr
        << "Test failure on Concurrent Pattern: Failed to create or allocate "
           "a ConfigurationBase object"
        << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int z_tests(int argc_, char **argv_) {
  asprintf(&argv_[2], "-z100");

  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->local_work_size != 100) {
    std::cerr << "Test failure on Run_Config Suite: -z with argument 100 had "
                 "incorrect value of "
              << cl1.configs[0]->local_work_size << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "-z500");

  Spatter::ClArgs cl2;
  if (parse_check(argc_, argv_, cl2) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl2.configs[0]->local_work_size != 500) {
    std::cerr << "Test failure on Run_Config Suite: -z with argument 500 had "
                 "incorrect value of "
              << cl2.configs[0]->local_work_size << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--local-work-size=1000");

  Spatter::ClArgs cl3;
  if (parse_check(argc_, argv_, cl3) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl3.configs[0]->local_work_size != 1000) {
    std::cerr << "Test failure on Run_Config Suite: -z with argument 1000 had "
                 "incorrect value of "
              << cl3.configs[0]->local_work_size << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int m_tests(int argc_, char **argv_) {
  asprintf(&argv_[2], "-m100");

  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->shmem != 100) {
    std::cerr << "Test Failure on Run_Config Suite: -m with argument 100 had "
                 "incorrect value of "
              << cl1.configs[0]->shmem << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "-m500");

  Spatter::ClArgs cl2;
  if (parse_check(argc_, argv_, cl2) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl2.configs[0]->shmem != 500) {
    std::cerr << "Test failure on Run_Config Suite: -m with argument 500 had "
                 "incorrect value of "
              << cl2.configs[0]->shmem << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--shared-mem=1000");

  Spatter::ClArgs cl3;
  if (parse_check(argc_, argv_, cl3) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl3.configs[0]->shmem != 1000) {
    std::cerr << "Test failure on Run_Config Suite: -m with argument 1000 had "
                 "incorrect value of "
              << cl3.configs[0]->shmem << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  int argc_ = 3;
  char **argv_ = (char **)malloc(sizeof(char *) * argc_);

  int ret;
  ret = asprintf(&argv_[0], "./spatter");
  if (ret == -1)
    return EXIT_FAILURE;

  ret = asprintf(&argv_[1], "-p1,2,3,4");
  if (ret == -1)
    return EXIT_FAILURE;

  // local-work-size z
  if (z_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // shared-mem m
  if (m_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  free(argv_[0]);
  free(argv_[1]);
  free(argv_);

  return EXIT_SUCCESS;
}
