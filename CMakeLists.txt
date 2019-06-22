cmake_minimum_required(VERSION 2.8.3)
project(starkit_fa)

# Build options
option(STARKIT_FA_BUILD_EXAMPLES "Build code examples" OFF)
option(STARKIT_FA_BUILD_VIEWER "Build FA Viewer (require SFML)" OFF)
option(STARKIT_FA_USES_DNN "Are deep neural networks available as function approximators (Strongly increases compilation time)" OFF)
option(STARKIT_FA_USES_GP "Are gaussian processes used as function approximators" OFF)

set (CATKIN_DEPS
  starkit_regression_forests
  starkit_bbo
)

if (STARKIT_FA_BUILD_VIEWER)
  set (CATKIN_DEPS ${CATKIN_DEPS} starkit_viewer)
endif(STARKIT_FA_BUILD_VIEWER)
if (STARKIT_FA_USES_DNN)
  set (CATKIN_DEPS ${CATKIN_DEPS} tiny-dnn)
  add_definitions(-DSTARKIT_FA_USES_DNN)
  # Use SSE2 and AVX if provided, otherwise, show a warning
  if ("${CMAKE_CXX_FLAGS}" MATCHES "-msse2" AND
      "${CMAKE_CXX_FLAGS}" MATCHES "-mavx")
    # Enable SSE2 optimizations
    add_definitions(-DCNN_USE_SSE)
    # Enable Advanced Vector Extensions
    add_definitions(-DCNN_USE_AVX)
  else()
    message(WARNING "missing flags for SSE2 and AVX! DNN will be really slow")
  endif()
endif (STARKIT_FA_USES_DNN)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})

set(DELEG_INCLUDE_DIRS include ${catkin_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17")

catkin_package(
  INCLUDE_DIRS ${DELEG_INCLUDE_DIRS}
  LIBRARIES starkit_fa
  CATKIN_DEPENDS ${CATKIN_DEPS}
  )

## Gaussian Processes support
# Currently, uses of GP in FA is directly linked with the use of GP in
# regression forests. It would be possible to separate those uses if necessary
# in the future.
if (STARKIT_FA_USES_GP)
  if (NOT "${catkin_LIBRARIES}" MATCHES "starkit_gp")
    message(FATAL_ERROR "starkit_fa cannot use starkit_gp because starkit_regression_forests does not uses starkit_gp")
  endif()
  add_definitions(-DSTARKIT_FA_USES_GP)
  add_definitions(-DSTARKIT_RF_USES_GP)
endif(STARKIT_FA_USES_GP)

###########
## Build ##
###########

include_directories(
  ${DELEG_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/starkit_fa/
  )

if (STARKIT_FA_BUILD_VIEWER)
  set(DIRECTORIES
    ${DIRECTORIES}
    src/starkit_fa/tools
    )
endif(STARKIT_FA_BUILD_VIEWER)

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
add_library(starkit_fa ${ALL_SOURCES} )
target_link_libraries(starkit_fa ${catkin_LIBRARIES} ${EXTRA_LIBS})

if (STARKIT_FA_BUILD_EXAMPLES)
  add_executable(test_fa_io src/test_io.cpp)
  target_link_libraries(test_fa_io starkit_fa ${DELEG_LIBRARIES}  )
  
  add_executable(test_optimizer_trainer src/test_optimizer_trainer.cpp)
  target_link_libraries(test_optimizer_trainer starkit_fa ${DELEG_LIBRARIES}  )
  
  add_executable(test_fa examples/test_fa.cpp)
  target_link_libraries(test_fa starkit_fa ${DELEG_LIBRARIES}  )
  
  add_executable(fa_on_dataset examples/fa_on_dataset.cpp)
  target_link_libraries(fa_on_dataset starkit_fa ${DELEG_LIBRARIES}  )
endif (STARKIT_FA_BUILD_EXAMPLES)

if (STARKIT_FA_BUILD_VIEWER)
  add_executable(viewer src/viewer.cpp)
  target_link_libraries(viewer starkit_fa ${DELEG_LIBRARIES}  )
endif(STARKIT_FA_BUILD_VIEWER)
