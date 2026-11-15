cmake_minimum_required(VERSION 3.18)
project(enceladus_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build against the interpreter that will import the package:
#   cmake -S cpp -B cpp/build -DPYBIND11_FINDPYTHON=ON -DPython_EXECUTABLE=$(which python3)
#   cmake --build cpp/build -j
set(PYBIND11_FINDPYTHON ON)

# Prefer a pip-installed pybind11 (python -m pybind11 --cmakedir); otherwise
# fetch it from GitHub.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE _pybind11_cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind11_rc
  ERROR_QUIET)
if(_pybind11_rc EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
  list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
endif()
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_enceladus_core src/bindings.cpp)
target_include_directories(_enceladus_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(_enceladus_core PRIVATE -O3)

# Drop the extension next to the Python package so `import enceladus_plume`
# picks it up via enceladus_plume/_native.py.
set_target_properties(_enceladus_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../enceladus_plume/enceladus_plume")
