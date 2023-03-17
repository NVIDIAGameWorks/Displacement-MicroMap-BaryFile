cmake_minimum_required(VERSION 3.15)

# Set the C/C++ specified in the projects as requirements
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(bary_core VERSION 1.0.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(UNIX)
  add_definitions(-DLINUX)
endif()

# XXXX we might be able to force this by setting CMAKE_GENERATOR_PLATFORM ??
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message( FATAL_ERROR "64-bit builds only : reconfigure with -A x64" )
endif()

set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")

#
# build
#

# If we need sanitizers enabled, add them here so that they apply to all
# bary_core projects.
# Note that libFuzzer is included in this list, because the fuzzer is a
# sanitizer, and emits instrumentation code to e.g. track the instruction
# counter. This instrumentation helps the fuzzer find new code paths.
option(BARY_CORE_BUILD_FUZZER "Enables the bary_fuzz target for fuzz-testing using libFuzzer. This should be off when creating a production build, for performance, since it emits program counter instrumentation code in bary_core." OFF)
set(BARY_CORE_USE_SANITIZERS "" CACHE STRING "Semicolon-separated list of LLVM sanitizers to compile with (example: address;memory). This should be empty when creating a production build, for performance.")
if(BARY_CORE_BUILD_FUZZER OR BARY_CORE_USE_SANITIZERS)
  set(_SANITIZERS ${BARY_CORE_USE_SANITIZERS})
  if(BARY_CORE_BUILD_FUZZER)
    list(APPEND _SANITIZERS "fuzzer")
  endif()
  if(MSVC)
    foreach(_SANITIZER IN ITEMS ${_SANITIZERS})
      add_compile_options("/fsanitize=${_SANITIZER}")
    endforeach()
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(JOIN _SANITIZERS "," _SANITIZERS_COMMA_JOINED)
    add_compile_options(-fsanitize=${_SANITIZERS_COMMA_JOINED} -fno-sanitize-recover=all)
    add_link_options(-fsanitize=${_SANITIZERS_COMMA_JOINED} -fno-sanitize-recover=all)
  endif()
endif()

# core lib

set(CORE_PUBLIC_HEADER_FILES
    include/bary/bary_api.h
    include/bary/bary_types.h
    include/bary/bary_core.h
)

set(CORE_SOURCE_FILES
    src/bary_core.cpp
)

source_group("public_include" FILES ${CORE_PUBLIC_HEADER_FILES})
source_group("source" FILES ${CORE_SOURCE_FILES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}")

add_library(bary_core STATIC ${CORE_SOURCE_FILES} ${CORE_PUBLIC_HEADER_FILES})

set_target_properties(bary_core PROPERTIES FOLDER "bary_core")

target_include_directories(bary_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)    

install(FILES ${CORE_PUBLIC_HEADER_FILES} DESTINATION "include")
install(TARGETS bary_core DESTINATION "lib" EXPORT "baryCoreTargets")
install(EXPORT "baryCoreTargets" FILE "baryCoreTargets.cmake" EXPORT_LINK_INTERFACE_LIBRARIES DESTINATION "cmake")


set(BARY_CORE_BUILD_TESTS OFF CACHE BOOL "Build the unit tests for bary_core")
set(BARY_CORE_BUILD_UTILS OFF CACHE BOOL "Build the utility library for bary_core")
set(BARY_CORE_BUILD_TOOL OFF CACHE BOOL "Build the tool for bary")

# utils lib

if (BARY_CORE_BUILD_UTILS OR BARY_CORE_BUILD_TOOL)
  set(UTILS_PUBLIC_HEADER_FILES
      include/baryutils/baryutils.h
  )

  set(UTILS_SOURCE_FILES
      src/bary_utils.cpp
  )

  source_group("public_include" FILES ${UTILS_PUBLIC_HEADER_FILES})
  source_group("source" FILES ${UTILS_SOURCE_FILES})

  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}")

  add_library(bary_utils STATIC ${UTILS_SOURCE_FILES} ${UTILS_PUBLIC_HEADER_FILES})

  set_target_properties(bary_utils PROPERTIES FOLDER "bary_utils")

  target_include_directories(bary_utils PUBLIC 
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)    

  install(FILES ${UTILS_PUBLIC_HEADER_FILES} DESTINATION "include")
  install(TARGETS bary_utils DESTINATION "lib" EXPORT "baryUtilsTargets")
  install(EXPORT "baryUtilsTargets" FILE "baryUtilsTargets.cmake" EXPORT_LINK_INTERFACE_LIBRARIES DESTINATION "cmake")
endif()

# tests

if(BARY_CORE_BUILD_TESTS)
    set(TEST_FILES test/main.cpp)
    source_group("tests" FILES ${TEST_FILES})
    add_executable(bary_core_test ${TEST_FILES})
    target_link_libraries(bary_core_test bary_core)
    set_target_properties(bary_core_test PROPERTIES FOLDER "bary_core")
endif()

# tool
if(BARY_CORE_BUILD_TOOL)
    set(TOOL_FILES 
      tool/filemapping.cpp
      tool/filemapping.hpp
      tool/bary_tool.cpp)
    source_group("tool" FILES ${TOOL_FILES})
    add_executable(bary_tool ${TOOL_FILES})
    target_link_libraries(bary_tool bary_core bary_utils)
    set_target_properties(bary_tool PROPERTIES FOLDER "bary_core")
endif()

# fuzzer
if(BARY_CORE_BUILD_FUZZER)
  set(FUZZER_FILES fuzz/bary_fuzz.cpp)
  source_group("fuzzer" FILES ${FUZZER_FILES})
  add_executable(bary_fuzzer ${FUZZER_FILES})
  target_link_libraries(bary_fuzzer bary_core)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CORPUS)
  set_target_properties(bary_fuzzer PROPERTIES
    FOLDER "bary_core"
    VS_DEBUGGER_COMMAND_ARGUMENTS "CORPUS -dict=${CMAKE_CURRENT_LIST_DIR}/fuzz/bary_fuzz.dict"
  )
endif()

#
# package target
#

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
include(CPack)
