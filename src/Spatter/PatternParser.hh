/*!
  \file PatternParser.hh
*/

#ifndef SPATTER_PATTERNPARSER_HH
#define SPATTER_PATTERNPARSER_HH

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Configuration.hh"
#include "SpatterTypes.hh"

namespace Spatter {

size_t power(size_t base, size_t exp);

int generate_pattern(std::string type,
    std::vector<std::vector<size_t>> generator,
    aligned_vector<size_t> &pattern);

int pattern_parser(
    std::stringstream &pattern_string, aligned_vector<size_t> &pattern);

} // namespace Spatter

#endif
