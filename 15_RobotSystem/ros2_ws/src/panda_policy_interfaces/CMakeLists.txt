cmake_minimum_required(VERSION 3.16)
project(panda_policy_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ResetEpisode.srv"
  "srv/StepEpisode.srv"
  "srv/ResetCartesian.srv"
  "srv/StepCartesian.srv"
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
