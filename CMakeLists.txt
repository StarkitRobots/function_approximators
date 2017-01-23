cmake_minimum_required(VERSION 2.8.3)
project(rosban_fa)

find_package(catkin REQUIRED COMPONENTS
  rosban_regression_forests
  rosban_bbo
)

find_package(rosban_viewer QUIET)

set(DELEG_INCLUDE_DIRS include ${catkin_INCLUDE_DIRS})

# Warning: Need to clean build in order to create the viewer if it was not
#          available during the first build
if (rosban_viewer_FOUND)
  message("rosban_viewer found, adding support for fa viewer")
  set (EXTRA_LIBS ${rosban_viewer_LIBRARIES})
  list(APPEND DELEG_INCLUDE_DIRS ${rosban_viewer_INCLUDE_DIRS})
  list(APPEND DELEG_LIBRARIES ${rosban_viewer_LIBRARIES})
else()
  message("rosban_viewer was not found, no support for fa viewer")
endif(rosban_viewer_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11")

catkin_package(
  INCLUDE_DIRS ${DELEG_INCLUDE_DIRS}
  LIBRARIES rosban_fa
  CATKIN_DEPENDS rosban_regression_forests rosban_bbo
)

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

if (rosban_viewer_FOUND)
  set(DIRECTORIES
    ${DIRECTORIES}
    src/rosban_fa/tools
    )
endif(rosban_viewer_FOUND)

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

if (rosban_viewer_FOUND)
  add_executable(viewer src/viewer.cpp)
  target_link_libraries(viewer rosban_fa ${DELEG_LIBRARIES}  )
endif(rosban_viewer_FOUND)
