#!/bin/sh
#This script initializes git submodules in a common script
#Each configure script should run this to check if modules have been initialized. If not, initialize them.

SUBMODULE_DIR=external/argtable3/src

check_submodule_init()
{
  #Check to see if the directory has been initialized



  if [ ! -d "$SUBMODULE_DIR" ]
  then
    printf "Pulling submodules\n"
    git submodule update --init --recursive
  else
    printf "Updating submodules\n"
    git submodule update --recursive
  fi

}

#Run the check and initialize, if needed
check_submodule_init
