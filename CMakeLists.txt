cmake_minimum_required(VERSION 2.8.3)
project(rosban_fa)

# Build options
option(BUILD_VIEWER "Build FA Viewer (require SFML)" OFF)

#TODO: add options for:
# - Use of BBO
# - Use of DNN
# - Building tests

find_package(catkin REQUIRED COMPONENTS
  rosban_regression_forests
  rosban_bbo
  tiny-dnn
)

set(DELEG_INCLUDE_DIRS include ${catkin_INCLUDE_DIRS})


if (BUILD_VIEWER)
  find_package(rosban_viewer REQUIRED)
  # Warning: Need to clean build in order to create the viewer if it was not
  #          available during the first build
  if (not rosban_viewer_FOUND)
    message("rosban_viewer was not found, impossible to build viewer ")
  endif(rosban_viewer_FOUND)
endif(BUILD_VIEWER)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11")

catkin_package(
  INCLUDE_DIRS ${DELEG_INCLUDE_DIRS}
  LIBRARIES rosban_fa
  CATKIN_DEPENDS rosban_regression_forests rosban_bbo
  )


# Use SSE2 and AVX if provided, otherwise, show a warning
#if ("${CMAKE_CXX_FLAGS}" MATCHES "-msse2" AND
#    "${CMAKE_CXX_FLAGS}" MATCHES "-mavx")
#  # Enable SSE2 optimizations
#  add_definitions(-DCNN_USE_SSE)
#  # Enable Advanced Vector Extensions
#  add_definitions(-DCNN_USE_AVX)
#else()
#  message(WARNING "missing flags for SSE2 and AVX! DNN will be really slow")
#endif()

###########
## Build ##
###########

include_directories(
  ${DELEG_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/rosban_fa/
  )

if (BUILD_VIEWER)
  set(DIRECTORIES
    ${DIRECTORIES}
    src/rosban_fa/tools
    )
endif(BUILD_VIEWER)

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

# Declare the library
add_library(rosban_fa ${ALL_SOURCES} )
target_link_libraries(rosban_fa ${catkin_LIBRARIES} ${EXTRA_LIBS})

add_executable(test_fa_io src/test_io.cpp)
target_link_libraries(test_fa_io rosban_fa ${DELEG_LIBRARIES}  )

add_executable(test_optimizer_trainer src/test_optimizer_trainer.cpp)
target_link_libraries(test_optimizer_trainer rosban_fa ${DELEG_LIBRARIES}  )

add_executable(test_fa examples/test_fa.cpp)
target_link_libraries(test_fa rosban_fa ${DELEG_LIBRARIES}  )

add_executable(fa_on_dataset examples/fa_on_dataset.cpp)
target_link_libraries(fa_on_dataset rosban_fa ${DELEG_LIBRARIES}  )

if (BUILD_VIEWER)
  add_executable(viewer src/viewer.cpp)
  target_link_libraries(viewer rosban_fa ${DELEG_LIBRARIES}  )
endif(BUILD_VIEWER)
