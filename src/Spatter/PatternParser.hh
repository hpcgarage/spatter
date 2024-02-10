/*!
  \file PatternParser.hh
*/

#ifndef SPATTER_PATTERNPARSER_HH
#define SPATTER_PATTERNPARSER_HH

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Spatter/Configuration.hh"
#include "Spatter/SpatterTypes.hh"

namespace Spatter {

size_t power(size_t base, size_t exp) {
  size_t result = 1;
  for (size_t i = 0; i < exp; ++i)
    result *= base;

  return result;
}

int generate_pattern(std::string type,
    std::vector<std::vector<size_t>> generator, aligned_vector<size_t> &pattern) {
  if (type.compare("UNIFORM") == 0) {
    if (generator.size() != 2) {
      std::cerr << "Parsing Error: Invalid UNIFORM Pattern "
                   "(UNIFORM:<length>:<stride>)"
                << std::endl;
      return -1;
    }

    size_t len = generator[0][0];
    size_t stride = generator[1][0];

    for (size_t i = 0; i < len; ++i)
      pattern.push_back(i * stride);
  } else if (type.compare("MS1") == 0) {
    if (generator.size() != 3) {
      std::cerr << "Parsing Error: Invalid MS1 Pattern "
                   "(MS1:<length>:<gap_locations>:<gap(s)>)"
                << std::endl;
      return -1;
    }

    size_t len;
    std::vector<size_t> gap_locations;
    std::vector<size_t> gaps;
    for (size_t i = 0; i < generator.size(); ++i) {
      if (i == 0)
        len = generator[i][0];
      if (i == 1)
        for (size_t j = 0; j < generator[i].size(); ++j)
          gap_locations.push_back(generator[i][j]);
      if (i == 2)
        for (size_t j = 0; j < generator[i].size(); ++j)
          gaps.push_back(generator[i][j]);
    }

    if (gap_locations.size() > 1 && gaps.size() != gap_locations.size() &&
        gaps.size() != 1) {
      std::cerr << "Parsing Error: Invalid MS1 Pattern "
                   "(MS1:<length>:<gap_locations>:<gap(s)>)"
                << std::endl;
      return -1;
    }

    size_t val = -1;
    size_t gap_index = 0;
    for (size_t i = 0; i < len; ++i) {
      if (gap_index < gap_locations.size() && gap_locations[gap_index] == i) {
        if (gaps.size() > 1)
          val += gaps[gap_index];
        else
          val += gaps[0];
        gap_index++;
      } else
        val++;

      pattern.push_back(val);
    }
  } else if (type.compare("LAPLACIAN") == 0) {
    if (generator.size() != 3) {
      std::cerr << "Parsing Error: Invalid LAPLACIAN Pattern "
                   "(LAPLACIAN:<dimension>:<pseudo_order>:<problem_size>)"
                << std::endl;
      return -1;
    }

    size_t dimension = generator[0][0];
    size_t pseudo_order = generator[1][0];
    size_t problem_size = generator[2][0];

    if (dimension < 1) {
      std::cerr << "Parsing Error: Invalid LAPLACIAN Pattern, Dimension must "
                   "be positive"
                << std::endl;
      return -1;
    }

    size_t final_len = dimension * pseudo_order * 2 + 1;
    size_t pos_len = 0;

    std::vector<size_t> pos;

    for (size_t i = 0; i < dimension; ++i) {
      for (size_t j = 0; j < pseudo_order; ++j) {
        pos.push_back((j + 1) * power(problem_size, i));
      }
      pos_len += pseudo_order;
    }

    size_t max = pos[pos_len - 1];

    for (size_t i = 0; i < final_len; ++i)
      pattern.push_back(2);

    for (size_t i = 0; i < pos_len; ++i)
      pattern[i] = -pos[pos_len - i - 1] + max;

    pattern[pos_len] = max;

    for (size_t i = 0; i < pos_len; ++i)
      pattern[pos_len + 1 + i] = pos[i] + max;

  } else {
    std::cerr << "Parsing Error: Invalid Pattern Generator Type (Valid types "
                 "are: UNIFORM, MS1, LAPLACIAN)"
              << std::endl;
    return -1;
  }

  return 0;
}

int pattern_parser(
    std::stringstream &pattern_string, aligned_vector<size_t> &pattern) {

  std::string type;
  std::vector<std::vector<size_t>> generator;

  if (pattern_string.str().rfind("UNIFORM", 0) == 0) {
    std::getline(pattern_string, type, ':');

    for (std::string line; std::getline(pattern_string, line, ':');) {
      try {
        size_t val = std::stoul(line);

        if (line[0] == '-') {
          std::cerr
              << "Parsing Error: Found Negative Index in Pattern Generator"
              << std::endl;
          return -1;
        } else {
          std::vector<size_t> values;
          values.push_back(val);
          generator.push_back(values);
        }
      } catch (const std::invalid_argument &ia) {
      }
    }
  } else if (pattern_string.str().rfind("MS1", 0) == 0) {
    std::getline(pattern_string, type, ':');

    for (std::string line; std::getline(pattern_string, line, ':');) {
      try {
        std::vector<size_t> values;

        std::stringstream linestream;
        linestream << line;
        for (std::string subline; std::getline(linestream, subline, ',');) {
          try {
            size_t val = std::stoul(subline);

            if (subline[0] == '-') {
              std::cerr << "Parsing Error: Found Negative Index in Pattern"
                        << std::endl;
              return -1;
            } else
              values.push_back(val);
          } catch (const std::invalid_argument &ia) {
            std::cerr << "Parsing Error: Invalid Pattern Generator Format"
                      << std::endl;
            return -1;
          }
        }
        generator.push_back(values);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Pattern Generator Format"
                  << std::endl;
        return -1;
      }
    }
  } else if (pattern_string.str().rfind("LAPLACIAN", 0) == 0) {
    std::getline(pattern_string, type, ':');

    for (std::string line; std::getline(pattern_string, line, ':');) {
      try {
        size_t val = std::stoul(line);

        if (line[0] == '-') {
          std::cerr
              << "Parsing Error: Found Negative Index in Pattern Generator"
              << std::endl;
          return -1;
        } else {
          std::vector<size_t> values;
          values.push_back(val);
          generator.push_back(values);
        }
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Pattern Generator Format"
                  << std::endl;
        return -1;
      }
    }
  }

  if (!type.empty())
    if (generate_pattern(type, generator, pattern) != 0)
      return -1;

  if (type.empty()) {
    for (std::string line; std::getline(pattern_string, line, ',');) {
      try {
        size_t val = std::stoul(line);

        if (line[0] == '-') {
          std::cerr << "Parsing Error: Found Negative Index in Pattern"
                    << std::endl;
          return -1;
        } else
          pattern.push_back(val);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Parsing Error: Invalid Pattern Format" << std::endl;
        return -1;
      }
    }
  }

  return 0;
}

} // namespace Spatter

#endif
