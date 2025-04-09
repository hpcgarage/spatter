/*!
  \file PatternParser.hh
*/

#ifndef SPATTER_PATTERNPARSER_HH
#define SPATTER_PATTERNPARSER_HH

#include <cstdint>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "Configuration.hh"
#include "SpatterTypes.hh"

#define CUSTOM_PATTERN "(^[0-9]+)(,[0-9]+)*$"
#define PAGE_BITS 12 // 12 bits => 4 KiB page

namespace Spatter {

size_t power(size_t base, size_t exp);

void compress_pattern(aligned_vector<size_t> &pattern);

int generate_pattern_custom(std::string pattern_string,
    aligned_vector<size_t> &pattern);

int generate_pattern_uniform(std::vector<std::string> args,
    aligned_vector<size_t> &pattern,
    size_t &delta);

int generate_pattern_ms1(std::vector<std::string> args,
    aligned_vector<size_t> &pattern);

int generate_pattern_laplacian(std::vector<std::string> args,
    aligned_vector<size_t> &pattern,
    size_t &delta);

int pattern_parser(std::stringstream &pattern_string,
    aligned_vector<size_t> &pattern,
    size_t &delta);

size_t remap_pattern(aligned_vector<size_t> &pattern,
    size_t &boundary,
    const size_t nrc);

int truncate_pattern(aligned_vector<size_t> &pattern, size_t pattern_size);

} // namespace Spatter

#endif
