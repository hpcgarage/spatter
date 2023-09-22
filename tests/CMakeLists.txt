list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
set( CMAKE_INCLUDE_CURRENT_DIR ON )
get_filename_component(full_path_json_test_data "${CMAKE_CURRENT_SOURCE_DIR}/test-data/amg.json" ABSOLUTE)
add_definitions(-DJSON_SRC="${full_path_json_test_data}")

get_filename_component(full_path_binary_test_folder "${CMAKE_CURRENT_SOURCE_DIR}/test-data/binary-traces" ABSOLUTE)
add_definitions(-DBINARY_TRACE_DIR="${full_path_binary_test_folder}")

set( TESTAPPS #list apps
        parse_uniform_stride_1
        parse_uniform_stride_suite
        parse_ms1_suite
        parse_custom_suite
        parse_omp_threads_suite
        parse_json_suite
        parse_run_config_suite
        parse_random_suite
        parse_concurrent
        parse_multilevel
        parse_size
        standard_uniform_suite
        standard_ms1_suite
        concurrent
        multilevel
        binary-trace
    )

IF("${BACKEND}" STREQUAL "cuda")
    set(TESTAPPS  ${TESTAPPS} standard_suite_gpu)
ELSE()
    set(TESTAPPS  ${TESTAPPS} standard_suite_cpu standard_laplacian_suite)
ENDIF()

file(GLOB src
    "${PROJECT_SOURCE_DIR}/external/argtable3/*.c"
    "${PROJECT_SOURCE_DIR}/src/*.c"
    "${PROJECT_SOURCE_DIR}/src/openmp/*.c"
    "${PROJECT_SOURCE_DIR}/src/cuda/*.c"
    "${PROJECT_SOURCE_DIR}/src/cuda/*.cu"
    "${PROJECT_SOURCE_DIR}/src/cuda/*.h"
    )
set( src_files ${src} )


#get full name of main.c
get_filename_component(full_path_main_c "${CMAKE_CURRENT_SOURCE_DIR}/../src/main.c" ABSOLUTE)
list(REMOVE_ITEM src_files "${full_path_main_c}")

foreach( APP ${TESTAPPS} )
 add_executable( "${APP}" EXCLUDE_FROM_ALL "${APP}.c" ${src_files})
 add_test("${APP}_build"
  "${CMAKE_COMMAND}"
  --build "${CMAKE_BINARY_DIR}"
  --config "$<CONFIG>"
  --target ${APP}
)
set_tests_properties("${APP}_build" PROPERTIES FIXTURES_SETUP "test_${APP}_fixture")
 target_link_libraries( "${APP}" PRIVATE m )
 IF ("${BACKEND}" STREQUAL "openmp")
     TARGET_LINK_LIBRARIES (${APP} PRIVATE OpenMP::OpenMP_CXX)
 ENDIF()
 add_test( NAME "${APP}_test" COMMAND "${APP}" )
 set_tests_properties("${APP}_test" PROPERTIES FIXTURES_REQUIRED "test_${APP}_fixture")
endforeach( APP ${TESTAPPS} )
