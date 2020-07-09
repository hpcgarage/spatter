#include <chrono>

using wall_clock_t = std::chrono::high_resolution_clock;
using time_point_t = std::chrono::time_point<wall_clock_t>;
template <typename T, class Period>
using time_interval_t = std::chrono::duration<T, Period>;

template <typename T>
class example_profiler {
  using event_list = std::vector<cl::sycl::event>;
  using time_point_list = std::vector<time_point_t>;

 public:
  struct profiling_data {
    T cgSubmissionTime{0};   // command group submission time
    T kernExecutionTime{0};  // exact computation time on the device
    T realExecutionTime{0};  // wall clock time
  };

  example_profiler() = default;
  example_profiler(event_list& events, const time_point_list& starts) {
    profile(events, starts);
  }

  void profile(event_list& eventList, const time_point_list& startTimeList) {
    if (startTimeList.size() != eventList.size()) {
      std::string errMsg =
          "The number of events do not match the number of starting time "
          "points.";
      throw std::runtime_error("Profiling Error:\n" + errMsg);
    }

    T cgSubmissionTime = 0;
    T kernExecutionTime = 0;
    T realExecutionTime = 0;
    
    const auto eventCount = eventList.size();
    for (size_t i = 0; i < eventCount; ++i) {
      auto curEvent = eventList.at(i);
      curEvent.wait();
      auto curStartTime = startTimeList.at(i);

      const auto end = wall_clock_t::now();
      time_interval_t<T, std::milli> curRealExecutionTime = end - curStartTime;
      realExecutionTime += curRealExecutionTime.count();

      const auto cgSubmissionTimePoint = curEvent.template get_profiling_info<
          cl::sycl::info::event_profiling::command_submit>();
      const auto startKernExecutionTimePoint =
          curEvent.template get_profiling_info<
              cl::sycl::info::event_profiling::command_start>();
      const auto endKernExecutionTimePoint =
          curEvent.template get_profiling_info<
              cl::sycl::info::event_profiling::command_end>();

      cgSubmissionTime +=
          to_milli(startKernExecutionTimePoint - cgSubmissionTimePoint);
      kernExecutionTime +=
          to_milli(endKernExecutionTimePoint - startKernExecutionTimePoint);
    }

    set_command_group_submission_time(cgSubmissionTime);
    set_kernel_execution_time(kernExecutionTime);
    set_real_execution_time(realExecutionTime);
  }

  inline T get_command_group_submission_time() const {
    return m_profData.cgSubmissionTime;
  }

  inline T get_kernel_execution_time() const {
    return m_profData.kernExecutionTime;
  }

  inline T get_real_execution_time() const {
    return m_profData.realExecutionTime;
  }

 private:
  profiling_data m_profData;

  inline void set_command_group_submission_time(T cgSubmissionTime) {
    m_profData.cgSubmissionTime = cgSubmissionTime;
  }

  inline void set_kernel_execution_time(T kernExecutionTime) {
    m_profData.kernExecutionTime = kernExecutionTime;
  }

  inline void set_real_execution_time(T realExecutionTime) {
    m_profData.realExecutionTime = realExecutionTime;
  }

  inline T to_milli(T timeValue) const {
    return timeValue * static_cast<T>(1e-6);
  }
};
