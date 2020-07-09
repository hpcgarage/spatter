list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
set( CMAKE_INCLUDE_CURRENT_DIR ON )
get_filename_component(full_path_json_test_data "${CMAKE_CURRENT_SOURCE_DIR}/test-data/amg.json" ABSOLUTE)
add_definitions(-DJSON_SRC="${full_path_json_test_data}")

if(USE_SYCL)

else()

set( TESTAPPS #list apps
        parse_uniform_stride_1
        parse_uniform_stride_suite
        parse_ms1_suite
        parse_custom_suite
        parse_omp_threads_suite
        parse_json_suite
        parse_run_config_suite
        parse_random_suite
    )

file(GLOB src
    "${PROJECT_SOURCE_DIR}/src/*.c"
    "${PROJECT_SOURCE_DIR}/src/openmp/*.c"
    )
set( src_files ${src} )


#get full name of main.c
get_filename_component(full_path_main_c "${CMAKE_CURRENT_SOURCE_DIR}/../src/main.c" ABSOLUTE)
list(REMOVE_ITEM src_files "${full_path_main_c}")

foreach( APP ${TESTAPPS} )
 add_executable( "${APP}" "${APP}.c" ${src_files})
 target_link_libraries( "${APP}" m )
 target_link_libraries( "${APP}" OpenMP::OpenMP_CXX)
 target_link_libraries("${APP}" argtable3_static)
 add_test( NAME "${APP}_test" COMMAND "${APP}" )
endforeach( APP ${TESTAPPS} )

endif()

