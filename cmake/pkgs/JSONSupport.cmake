include(FetchContent)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz OVERRIDE_FIND_PACKAGE)
FetchContent_MakeAvailable(nlohmann_json)

find_package(nlohmann_json 3.11.2 REQUIRED)
set(COMMON_LINK_LIBRARIES ${COMMON_LINK_LIBRARIES} nlohmann_json::nlohmann_json)
