# Spatter/test/CMakelists.txt
get_filename_component(full_path_json_test_data "${CMAKE_CURRENT_SOURCE_DIR}/test-data/amg.json" ABSOLUTE)
add_definitions(-DJSON_SRC="${full_path_json_test_data}")

get_filename_component(full_path_binary_test_folder "${CMAKE_CURRENT_SOURCE_DIR}/test-data/binary-traces" ABSOLUTE)
add_definitions(-DBINARY_TRACE_DIR="${full_path_binary_test_folder}")

set(SPATTER_TESTS
      parse_uniform_stride_1
      parse_uniform_stride_suite
      parse_ms1_suite
      parse_custom_suite
      parse_json_suite
      parse_run_config_suite_cpu
      parse_random_suite
      parse_concurrent
      parse_multilevel
      parse_size
      standard_uniform_suite
      standard_ms1_suite
      concurrent
      multilevel
      binary-trace
      standard_suite_cpu
      standard_laplacian_suite
  )

if (USE_OPENMP)
  set(SPATTER_TESTS ${SPATTER_TESTS} parse_omp_threads_suite)
endif()

if (USE_CUDA)
  set(SPATTER_TESTS ${SPATTER_TESTS} standard_suite_gpu parse_run_config_suite_gpu)
endif()

foreach (APP ${SPATTER_TESTS})
  add_executable(${APP} ${APP}.cc)
  target_link_libraries(${APP} Spatter)
  target_include_directories(${APP} PUBLIC ${PROJECT_SOURCE_DIR}/src)

  add_test(NAME "${APP}_build" COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target ${APP})
  set_tests_properties("${APP}_build" PROPERTIES FIXTURES_SETUP "test_${APP}_fixture")

  add_test(NAME "${APP}_test" COMMAND "${APP}")
  set_tests_properties("${APP}_test" PROPERTIES FIXTURES_REQUIRED "test_${APP}_fixture")
endforeach (APP ${SPATTER_TESTS})

