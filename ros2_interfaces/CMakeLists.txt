cmake_minimum_required(VERSION 3.5)
project(shg_interfaces)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(DEPENDENCIES
  builtin_interfaces
  std_msgs
  geometry_msgs
)

#file (GLOB MSGS msg/*.msg)
#file (GLOB ACTIONS action/*.action)
#file (GLOB SERVICES srv/*.srv)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetElevatorData.srv"

  #${MSGS}
  #${ACTIONS}
  #${SERVICES}
  DEPENDENCIES ${DEPENDENCIES})

ament_export_dependencies(rosidl_default_runtime ${DEPENDENCIES})
ament_package()
