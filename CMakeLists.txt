project(test_kaze_port)
cmake_minimum_required(VERSION 2.4)

# ============================================================================ #
add_definitions(-DDEBUG -g -O3 -Wall -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE)
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# ============================================================================ #
# OpenCV
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})

include_directories(
   ${OpenCV_INCLUDE_DIRS}
)

link_directories(
   ${OpenCV_LIBRARY_DIRS}
)

# ============================================================================ #
set(SRCS
  ./src/utils.cpp
)
# ============================================================================ #
# KAZE tests
add_executable(
  test_kaze
  test_kaze.cpp
  ${SRCS}
)

target_link_libraries(
  test_kaze
  ${OpenCV_LIBS}
)

add_executable(
  test_kaze_match
  test_kaze_match.cpp
  ${SRCS}
)

target_link_libraries(
  test_kaze_match
  ${OpenCV_LIBS}
)

# ============================================================================ #
# A-KAZE tests
add_executable(
  test_akaze
  test_akaze.cpp
  ${SRCS}
)

target_link_libraries(
  test_akaze
  ${OpenCV_LIBS}
)

add_executable(
  test_akaze_match
  test_akaze_match.cpp
  ${SRCS}
)

target_link_libraries(
  test_akaze_match
  ${OpenCV_LIBS}
)
