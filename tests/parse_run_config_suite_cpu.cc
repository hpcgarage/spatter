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

int k_tests(int argc_, char **argv_) {
  int sg_argc_ = 4;
  char **sg_argv_ = (char **)malloc(sizeof(char *) * sg_argc_);

  int ret;
  ret = asprintf(&sg_argv_[0], "./spatter");
  if (ret == -1)
    return EXIT_FAILURE;

  ret = asprintf(&sg_argv_[1], "-g1,2,3,4");
  if (ret == -1)
    return EXIT_FAILURE;

  ret = asprintf(&sg_argv_[2], "-u1,2,3,4");
  if (ret == -1)
    return EXIT_FAILURE;

  asprintf(&argv_[2], "-kGATHER");
  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->kernel.compare("gather") != 0) {
    std::cerr << "Test failure on Run_Config Suite: POSIX-style k with "
                 "argument GATHER resulted in kernel "
              << cl1.configs[0]->kernel << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "-kSCATTER");

  Spatter::ClArgs cl2;
  if (parse_check(argc_, argv_, cl2) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl2.configs[0]->kernel.compare("scatter") != 0) {
    std::cerr << "Test failure on Run_Config Suite: POSIX-style k with "
                 "argument SCATTER resulted in kernel "
              << cl2.configs[0]->kernel << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&sg_argv_[3], "-kSG");

  Spatter::ClArgs cl3;
  if (parse_check(sg_argc_, sg_argv_, cl3) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(sg_argv_[3]);

  if (cl3.configs[0]->kernel.compare("sg") != 0) {
    std::cerr << "Test failure on Run_Config Suite: POSIX-style k with "
                 "argument GS resulted in kernel "
              << cl3.configs[0]->kernel << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--kernel=GATHER");

  Spatter::ClArgs cl4;
  if (parse_check(argc_, argv_, cl4) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl4.configs[0]->kernel.compare("gather") != 0) {
    std::cerr << "Test failure on Run_Config Suite: --kernel with argument "
                 "GATHER resulted in kernel "
              << cl4.configs[0]->kernel << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--kernel=SCATTER");

  Spatter::ClArgs cl5;
  if (parse_check(argc_, argv_, cl5) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl5.configs[0]->kernel.compare("scatter") != 0) {
    std::cerr << "Test failure on Run_Config Suite: --kernel with argument "
                 "SCATTER resulted in kernel "
              << cl5.configs[0]->kernel << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&sg_argv_[3], "--kernel=SG");

  Spatter::ClArgs cl6;
  if (parse_check(sg_argc_, sg_argv_, cl6) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(sg_argv_[3]);

  if (cl6.configs[0]->kernel.compare("sg") != 0) {
    std::cerr << "Test failure on Run_Config Suite: --kernel with argument SG "
                 "resulted in kernel "
              << cl6.configs[0]->kernel << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int d_tests(int argc_, char **argv_) {
  asprintf(&argv_[2], "-d9");

  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->delta != 9) {
    std::cerr << "Test failure on Run_Config Suite: -d with argument 9 had "
                 "incorrect element of "
              << cl1.configs[0]->delta << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int l_tests(int argc_, char **argv_) {
  asprintf(&argv_[2], "-l100");

  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->count != 100) {
    std::cerr << "Test failure on Run_Config Suite: -l with argument 100 had "
                 "incorrect value of "
              << cl1.configs[0]->count << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "-l500");

  Spatter::ClArgs cl2;
  if (parse_check(argc_, argv_, cl2) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl2.configs[0]->count != 500) {
    std::cerr << "Test failure on Run_Config Suite: -l with argument 500 had "
                 "incorrect value of "
              << cl2.configs[0]->count << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--count=1000");

  Spatter::ClArgs cl3;
  if (parse_check(argc_, argv_, cl3) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl3.configs[0]->count != 1000) {
    std::cerr << "Test failure on Run_Config Suite: -l with argument 1000 had "
                 "incorrect value of "
              << cl3.configs[0]->count << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int w_tests(int argc_, char **argv_) {
  asprintf(&argv_[2], "-w100");

  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->wrap != 100) {
    std::cerr << "Test failure on Run_Config Suite: -w with argument 100 had "
                 "incorrect value of "
              << cl1.configs[0]->wrap << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "-w500");

  Spatter::ClArgs cl2;
  if (parse_check(argc_, argv_, cl2) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl2.configs[0]->wrap != 500) {
    std::cerr << "Test failure on Run_Config Suite: -w with argument 500 had "
                 "incorrect value of "
              << cl2.configs[0]->wrap << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--wrap=1000");

  Spatter::ClArgs cl3;
  if (parse_check(argc_, argv_, cl3) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl3.configs[0]->wrap != 1000) {
    std::cerr << "Test failure on Run_Config Suite: -w with argument 1000 had "
                 "incorrect value of "
              << cl3.configs[0]->wrap << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int r_tests(int argc_, char **argv_) {
  asprintf(&argv_[2], "-r100");

  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->nruns != 100) {
    std::cerr << "Test failure on Run_Config Suite: -r with argument 100 had "
                 "incorrect value of "
              << cl1.configs[0]->nruns << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "-r500");

  Spatter::ClArgs cl2;
  if (parse_check(argc_, argv_, cl2) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl2.configs[0]->nruns != 500) {
    std::cerr << "Test failure on Run_Config Suite: -r with argument 500 had "
                 "incorrect value of "
              << cl2.configs[0]->nruns << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--runs=1000");

  Spatter::ClArgs cl3;
  if (parse_check(argc_, argv_, cl3) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl3.configs[0]->nruns != 1000) {
    std::cerr << "Test failure on Run_Config Suite: -r with argument 1000 had "
                 "incorrect value of "
              << cl3.configs[0]->nruns << "." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int n_tests(int argc_, char **argv_) {
  asprintf(&argv_[2], "-nTestName");

  Spatter::ClArgs cl1;
  if (parse_check(argc_, argv_, cl1) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl1.configs[0]->name.compare("TestName") != 0) {
    std::cerr << "Test failure on Run_Config Suite: -n with argument TestName "
                 "had incorrect value of "
              << cl1.configs[0]->name << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "-nTestName2");

  Spatter::ClArgs cl2;
  if (parse_check(argc_, argv_, cl2) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl2.configs[0]->name.compare("TestName2") != 0) {
    std::cerr << "Test failure on Run_Config Suite: -n with argument TestName2 "
                 "had incorrect value of "
              << cl2.configs[0]->name << "." << std::endl;
    return EXIT_FAILURE;
  }

  asprintf(&argv_[2], "--name=TestName3");

  Spatter::ClArgs cl3;
  if (parse_check(argc_, argv_, cl3) == EXIT_FAILURE)
    return EXIT_FAILURE;

  free(argv_[2]);

  if (cl3.configs[0]->name.compare("TestName3") != 0) {
    std::cerr << "Test failure on Run_Config Suite: -n with argument TestName3 "
                 "had incorrect value of "
              << cl3.configs[0]->name << "." << std::endl;
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

  // kernel-name k
  if (k_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // delta d
  if (d_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // count l
  if (l_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // wrap w
  if (w_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // runs r
  if (r_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // name n
  if (n_tests(argc_, argv_) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  free(argv_[0]);
  free(argv_[1]);
  free(argv_);

  return EXIT_SUCCESS;
}
