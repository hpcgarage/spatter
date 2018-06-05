/** @file parse-args.h
 *  @author Patrick Lavin
 *  @brief Provides a function to read CLI 
 */

#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

#define STRING_SIZE 100

/** @brief Read command-line arguments and populate global variables. 
 *  @param argc Value passed to main
 *  @param argv Value passed to main
 */
void parse_args(int argc, char **argv);

/** @brief Supported benchmark backends
 */
enum sg_backend
{
    OPENCL, /** OpenCL Backend */
    OPENMP, /** OpenMP CPU Backend */
    INVALID /** Used as a default backend */
};

#endif 
