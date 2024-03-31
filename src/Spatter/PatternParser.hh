/*!
  \file PatternParser.hh
*/

#ifndef SPATTER_PATTERNPARSER_HH
#define SPATTER_PATTERNPARSER_HH

#include <cstdint>
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
    std::stringstream &pattern_string, aligned_vector<size_t> &pattern, size_t &delta);

size_t remap_pattern(aligned_vector<size_t> &pattern, const size_t boundary);

int truncate_pattern(aligned_vector<size_t> &pattern, size_t pattern_size);

} // namespace Spatter

#endif
