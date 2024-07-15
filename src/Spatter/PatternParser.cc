/*!
  \file PatternParser.cc
*/

#include "PatternParser.hh"

namespace Spatter {

size_t power(size_t base, size_t exp) {
  size_t result = 1;
  for (size_t i = 0; i < exp; ++i)
    result *= base;

  return result;
}

void compress_pattern(aligned_vector<size_t> &pattern) {
  std::vector<long> pages;

  const size_t pattern_len = pattern.size();
  for (size_t i = 0; i < pattern_len; i++) {
    size_t page = (pattern[i] * 8) >> PAGE_BITS;
    size_t page_index;

    auto it = std::find(pages.begin(), pages.end(), page);
    if (it != pages.end()) {
      page_index = it - pages.begin();
    } else {
      page_index = pages.size();
      pages.push_back(page);
    }

    size_t new_val = (page_index << PAGE_BITS) |
        ((pattern[i] * 8) & ((1l << PAGE_BITS) - 1l));
    new_val /= 8;
    pattern[i] = new_val;
  }
}

int generate_pattern_uniform(std::vector<std::string> args,
    aligned_vector<size_t> &pattern,
    size_t &delta) {
  if ((args.size() != 2) && (args.size() != 3)) {
    std::cerr << "Parsing Error: Invalid UNIFORM Pattern "
                   "(UNIFORM:<length>:<stride>[:<delta|NR>])"
                << std::endl;
    return -1;
  }

  int64_t length, stride, new_delta;

  try {
    length = std::stoll(args[0]);
    stride = std::stoll(args[1]);

    if (length < 1)
      throw std::invalid_argument("Invalid length");

    if (stride < 1)
      throw std::invalid_argument("Invalid stride");

    if (args.size() == 3) {
      if (args[2].compare("NR") == 0) {
        delta = length * stride;
      } else {
        new_delta = std::stoll(args[2]);

        if (new_delta < 1)
          throw std::invalid_argument("Invalid delta");
        else
          delta = new_delta;
      }
    }
  } catch (const std::invalid_argument &ia) {
    std::cerr << "Parsing Error: Invalid UNIFORM Pattern "
                  "(UNIFORM:<length>:<stride>[:<delta|NR>])"
              << std::endl;
    return -1;
  }

  for (int64_t i = 0; i < length; ++i)
    pattern.push_back(i * stride);

  return 0;
}

int generate_pattern_ms1(std::vector<std::string> args,
    aligned_vector<size_t> &pattern) {
    // if (generator.size() != 3) {
    //   std::cerr << "Parsing Error: Invalid MS1 Pattern "
    //                "(MS1:<length>:<gap_locations>:<gap(s)>)"
    //             << std::endl;
    //   return -1;
    // }

    // size_t len = 0;
    // std::vector<size_t> gap_locations;
    // std::vector<size_t> gaps;
    // for (size_t i = 0; i < generator.size(); ++i) {
    //   if (i == 0)
    //     len = generator[i][0];
    //   if (i == 1)
    //     for (size_t j = 0; j < generator[i].size(); ++j)
    //       gap_locations.push_back(generator[i][j]);
    //   if (i == 2)
    //     for (size_t j = 0; j < generator[i].size(); ++j)
    //       gaps.push_back(generator[i][j]);
    // }

    // if (gap_locations.size() > 1 && gaps.size() != gap_locations.size() &&
    //    td::cerr << "Parsing Error: Invalid MS1 Pattern "
    //                "(MS1:<length>:<gap_locations>:<gap(s)>)"
    //             << std::endl;
    //   return -1;
    // }

    // int64_t val = -1;
    // size_t gap_index = 0;
    // for (size_t i = 0; i < len; ++i) {
    //   if (gap_index < gap_locations.size() && gap_locations[gap_index] == i) {
    //     if (gaps.size() > 1)
    //       val += gaps[gap_index];
    //     else
    //       val += gaps[0];
    //     gap_index++;
    //   } else
    //     val++;

    //   pattern.push_back(static_cast<size_t>(val));
    // }
    // return 0; gaps.size() != 1) {
    //   std::cerr << "Parsing Error: Invalid MS1 Pattern "
    //                "(MS1:<length>:<gap_locations>:<gap(s)>)"
    //             << std::endl;
    //   return -1;
    // }

    // int64_t val = -1;
    // size_t gap_index = 0;
    // for (size_t i = 0; i < len; ++i) {
    //   if (gap_index < gap_locations.size() && gap_locations[gap_index] == i) {
    //     if (gaps.size() > 1)
    //       val += gaps[gap_index];
    //     else
    //       val += gaps[0];
    //     gap_index++;
    //   } else
    //     val++;

    //   pattern.push_back(static_cast<size_t>(val));
    // }
    return 0;
}

int generate_pattern_laplacian(std::vector<std::string> args,
    aligned_vector<size_t> &pattern) {
    // if (generator.size() != 3) {
    //   std::cerr << "Parsing Error: Invalid LAPLACIAN Pattern "
    //                "(LAPLACIAN:<dimension>:<pseudo_order>:<problem_size>)"
    //             << std::endl;
    //   return -1;
    // }

    // size_t dimension = generator[0][0];
    // size_t pseudo_order = generator[1][0];
    // size_t problem_size = generator[2][0];

    // if (dimension < 1) {
    //   std::cerr << "Parsing Error: Invalid LAPLACIAN Pattern, Dimension must "
    //                "be positive"
    //             << std::endl;
    //   return -1;
    // }

    // size_t final_len = dimension * pseudo_order * 2 + 1;
    // size_t pos_len = 0;

    // std::vector<size_t> pos;

    // for (size_t i = 0; i < dimension; ++i) {
    //   for (size_t j = 0; j < pseudo_order; ++j) {
    //     pos.push_back((j + 1) * power(problem_size, i));
    //   }
    //   pos_len += pseudo_order;
    // }

    // size_t max = pos[pos_len - 1];

    // for (size_t i = 0; i < final_len; ++i)
    //   pattern.push_back(2);

    // for (size_t i = 0; i < pos_len; ++i)
    //   pattern[i] = -pos[pos_len - i - 1] + max;

    // pattern[pos_len] = max;

    // for (size_t i = 0; i < pos_len; ++i)
    //   pattern[pos_len + 1 + i] = pos[i] + max;

    return 0;
}

int pattern_parser(std::stringstream &pattern_string,
    aligned_vector<size_t> &pattern,
    size_t &delta) {
  std::string type, line;
  std::vector<std::string> args;

  std::getline(pattern_string, type, ':');

  while (std::getline(pattern_string, line, ':'))
    args.push_back(line);

  int ret = -1;

  if (type.compare("UNIFORM") == 0)
    ret = generate_pattern_uniform(args, pattern, delta);
  else if (type.compare("MS1") == 0)
    ret = generate_pattern_ms1(args, pattern);
  else if (type.compare("LAPLACIAN") == 0)
    ret = generate_pattern_laplacian(args, pattern);
  else
    std::cerr << "Parsing Error: Invalid Pattern Generator Type (Valid types "
                 "are: UNIFORM, MS1, LAPLACIAN)"
              << std::endl << "Recieved: " << type << std::endl;

  return ret;
}

size_t remap_pattern(aligned_vector<size_t> &pattern, const size_t boundary) {
  const size_t pattern_len = pattern.size();
  for (size_t j = 0; j < pattern_len; ++j) {
    pattern[j] = pattern[j] % boundary;
  }

  size_t max_pattern_val = *(std::max_element(pattern.begin(), pattern.end()));
  return max_pattern_val;
}

int truncate_pattern(aligned_vector<size_t> &pattern, size_t pattern_size) {
  if (pattern_size > pattern.size()) {
    return -1;
  }

  pattern.resize(pattern_size);

  return 0;
}

} // namespace Spatter
