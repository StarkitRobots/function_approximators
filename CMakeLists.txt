cmake_minimum_required(VERSION 2.8.3)
project(rhoban_fa)

# Build options
option(RHOBAN_FA_BUILD_EXAMPLES "Build code examples" OFF)
option(RHOBAN_FA_BUILD_VIEWER "Build FA Viewer (require SFML)" OFF)
option(RHOBAN_FA_USES_DNN "Are deep neural networks available as function approximators (Strongly increases compilation time)" OFF)
option(RHOBAN_FA_USES_GP "Are gaussian processes used as function approximators" OFF)

set (CATKIN_DEPS
  rhoban_regression_forests
  rhoban_bbo
)

if (RHOBAN_FA_BUILD_VIEWER)
  set (CATKIN_DEPS ${CATKIN_DEPS} rhoban_viewer)
endif(RHOBAN_FA_BUILD_VIEWER)
if (RHOBAN_FA_USES_DNN)
  set (CATKIN_DEPS ${CATKIN_DEPS} tiny-dnn)
  add_definitions(-DRHOBAN_FA_USES_DNN)
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
endif (RHOBAN_FA_USES_DNN)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})

set(DELEG_INCLUDE_DIRS include ${catkin_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11")

catkin_package(
  INCLUDE_DIRS ${DELEG_INCLUDE_DIRS}
  LIBRARIES rhoban_fa
  CATKIN_DEPENDS ${CATKIN_DEPS}
  )

## Gaussian Processes support
# Currently, uses of GP in FA is directly linked with the use of GP in
# regression forests. It would be possible to separate those uses if necessary
# in the future.
if (RHOBAN_FA_USES_GP)
  message("CATKIN_LIBS: ${catkin_LIBRARIES}")
  message("IDX: ${IDX}")
  if (NOT "${catkin_LIBRARIES}" MATCHES "rhoban_gp")
    message(FATAL_ERROR "rhoban_fa cannot use rhoban_gp because rhoban_regression_forests does not uses rhoban_gp")
  endif()
  add_definitions(-DRHOBAN_FA_USES_GP)
  add_definitions(-DRHOBAN_RF_USES_GP)
endif(RHOBAN_FA_USES_GP)

###########
## Build ##
###########

include_directories(
  ${DELEG_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/rhoban_fa/
  )

if (RHOBAN_FA_BUILD_VIEWER)
  set(DIRECTORIES
    ${DIRECTORIES}
    src/rhoban_fa/tools
    )
endif(RHOBAN_FA_BUILD_VIEWER)

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
add_library(rhoban_fa ${ALL_SOURCES} )
target_link_libraries(rhoban_fa ${catkin_LIBRARIES} ${EXTRA_LIBS})

if (RHOBAN_FA_BUILD_EXAMPLES)
  add_executable(test_fa_io src/test_io.cpp)
  target_link_libraries(test_fa_io rhoban_fa ${DELEG_LIBRARIES}  )
  
  add_executable(test_optimizer_trainer src/test_optimizer_trainer.cpp)
  target_link_libraries(test_optimizer_trainer rhoban_fa ${DELEG_LIBRARIES}  )
  
  add_executable(test_fa examples/test_fa.cpp)
  target_link_libraries(test_fa rhoban_fa ${DELEG_LIBRARIES}  )
  
  add_executable(fa_on_dataset examples/fa_on_dataset.cpp)
  target_link_libraries(fa_on_dataset rhoban_fa ${DELEG_LIBRARIES}  )
endif (RHOBAN_FA_BUILD_EXAMPLES)

if (RHOBAN_FA_BUILD_VIEWER)
  add_executable(viewer src/viewer.cpp)
  target_link_libraries(viewer rhoban_fa ${DELEG_LIBRARIES}  )
endif(RHOBAN_FA_BUILD_VIEWER)
