#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#include "Configuration.hh"
#include <chrono>

#include <cmath>

#include <algorithm>

void cuda_gather(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  double x;

  if (i < count) {
    // dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];
    x = sparse[pattern[j] + delta * i];
    if (x == 0.5)
      dense[0] = x;
  }
}

void cuda_scatter(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];
}

void cuda_scatter_atomic(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    dpct::atomic_exchange<unsigned long long,
        sycl::access::address_space::generic_space>(
        (unsigned long long int *)&sparse[pattern[j] + delta * i],
        (unsigned long long)(sycl::bit_cast<long long>(
            dense[j + pattern_length * (i % wrap)])));
}

void cuda_scatter_gather(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  // printf("%lu, %lu, %lu\n", total_id, j, i);
  if (i < count)
    sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
        sparse_gather[pattern_gather[j] + delta_gather * i];
}

void cuda_scatter_gather_atomic(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  // printf("%lu, %lu, %lu\n", total_id, j, i);
  if (i < count)
    dpct::atomic_exchange<unsigned long long,
        sycl::access::address_space::generic_space>(
        (unsigned long long int
                *)&sparse_scatter[pattern_scatter[j] + delta_scatter * i],
        (unsigned long long)(sycl::bit_cast<long long>(
            sparse_gather[pattern_gather[j] + delta_gather * i])));
}

void cuda_multi_gather(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  double x;

  if (i < count) {
    // dense[j + pattern_length * (i % wrap)] =
    // sparse[pattern[pattern_gather[j]] + delta * i];
    x = sparse[pattern[pattern_gather[j]] + delta * i];
    if (x == 0.5)
      dense[0] = x;
  }
}

void cuda_multi_scatter(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    sparse[pattern[pattern_scatter[j]] + delta * i] =
        dense[j + pattern_length * (i % wrap)];
}

void cuda_multi_scatter_atomic(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count, sycl::nd_item<3> item_ct1) {
  size_t total_id = (size_t)((size_t)item_ct1.get_local_range(2) *
          (size_t)item_ct1.get_group(2) +
      (size_t)item_ct1.get_local_id(2));
  size_t j = total_id % pattern_length; // pat_idx
  size_t i = total_id / pattern_length; // count_idx

  if (i < count)
    dpct::atomic_exchange<unsigned long long,
        sycl::access::address_space::generic_space>(
        (unsigned long long int
                *)&sparse[pattern[pattern_scatter[j]] + delta * i],
        (unsigned long long)(sycl::bit_cast<long long>(
            dense[j + pattern_length * (i % wrap)])));
}

float cuda_gather_wrapper(const size_t *pattern, const double *sparse,
    double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:8: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_gather(pattern, sparse, dense, pattern_length, delta, wrap, count,
            item_ct1);
      });
  /*
  DPCT1026:9: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:10: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}

float cuda_scatter_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:11: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_scatter(pattern, sparse, dense, pattern_length, delta, wrap, count,
            item_ct1);
      });
  /*
  DPCT1026:12: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:13: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}

float cuda_scatter_atomic_wrapper(const size_t *pattern, double *sparse,
    const double *dense, const size_t pattern_length, const size_t delta,
    const size_t wrap, const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:14: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_scatter_atomic(pattern, sparse, dense, pattern_length, delta, wrap,
            count, item_ct1);
      });
  /*
  DPCT1026:15: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:16: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}

float cuda_scatter_gather_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:17: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_scatter_gather(pattern_scatter, sparse_scatter, pattern_gather,
            sparse_gather, pattern_length, delta_scatter, delta_gather, wrap,
            count, item_ct1);
      });
  /*
  DPCT1026:18: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:19: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}

float cuda_scatter_gather_atomic_wrapper(const size_t *pattern_scatter,
    double *sparse_scatter, const size_t *pattern_gather,
    const double *sparse_gather, const size_t pattern_length,
    const size_t delta_scatter, const size_t delta_gather, const size_t wrap,
    const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:20: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_scatter_gather_atomic(pattern_scatter, sparse_scatter,
            pattern_gather, sparse_gather, pattern_length, delta_scatter,
            delta_gather, wrap, count, item_ct1);
      });
  /*
  DPCT1026:21: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:22: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}

float cuda_multi_gather_wrapper(const size_t *pattern,
    const size_t *pattern_gather, const double *sparse, double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:23: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_multi_gather(pattern, pattern_gather, sparse, dense,
            pattern_length, delta, wrap, count, item_ct1);
      });
  /*
  DPCT1026:24: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:25: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}

float cuda_multi_scatter_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:26: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_multi_scatter(pattern, pattern_scatter, sparse, dense,
            pattern_length, delta, wrap, count, item_ct1);
      });
  /*
  DPCT1026:27: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:28: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}

float cuda_multi_scatter_atomic_wrapper(const size_t *pattern,
    const size_t *pattern_scatter, double *sparse, const double *dense,
    const size_t pattern_length, const size_t delta, const size_t wrap,
    const size_t count) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start = new sycl::event();
  stop = new sycl::event();

  int threads_per_block = std::min(pattern_length, (size_t)1024);
  int blocks_per_grid =
      ((pattern_length * count) + threads_per_block - 1) / threads_per_block;

  dev_ct1.queues_wait_and_throw();
  /*
  DPCT1012:29: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, blocks_per_grid) *
              sycl::range<3>(1, 1, threads_per_block),
          sycl::range<3>(1, 1, threads_per_block)),
      [=](sycl::nd_item<3> item_ct1) {
        cuda_multi_scatter_atomic(pattern, pattern_scatter, sparse, dense,
            pattern_length, delta, wrap, count, item_ct1);
      });
  /*
  DPCT1026:30: The call to cudaGetLastError was removed because this call is
  redundant in SYCL.
  */

  /*
  DPCT1012:31: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();

  float time_ms = 0;
  time_ms =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  dpct::destroy_event(start);
  dpct::destroy_event(stop);

  return time_ms;
}
