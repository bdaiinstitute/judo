cmake_minimum_required(VERSION 3.15)
project(judo_cpp)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

# 1. Locate Python, which is used to find the MuJoCo package
find_package(Python
  COMPONENTS
    Interpreter
    Development.Module
  REQUIRED
)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
  OUTPUT_VARIABLE MJ_PKG_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "MuJoCo package root: ${MJ_PKG_ROOT}")

# 2. Glob for the actual libmujoco.so file(s) there
file(GLOB MUJOCO_LIB_FILES
     "${MJ_PKG_ROOT}/libmujoco.so*"  # Linux
     "${MJ_PKG_ROOT}/libmujoco.dylib"  # macOS default
     "${MJ_PKG_ROOT}/libmujoco.*.dylib"  # macOS versioned dylib
)
if(NOT MUJOCO_LIB_FILES)
  message(FATAL_ERROR "Could not find any libmujoco.so* in ${MJ_PKG_ROOT}")
endif()
list(GET MUJOCO_LIB_FILES 0 MUJOCO_LIB)
message(STATUS "Linking against MuJoCo library: ${MUJOCO_LIB}")

# 3. Pull in mujoco headers - they're located under the include directory
set(MJINC_DIR "${MJ_PKG_ROOT}/include")
message(STATUS "MuJoCo include dir: ${MJINC_DIR}")
include_directories(${MJINC_DIR})

# 4. Build the pybind11 module
find_package(pybind11 REQUIRED)
pybind11_add_module(_judo_cpp
  bindings.cpp
  rollout.cpp
  spot_rollout.cpp
)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE SITE_PACKAGES_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)  # find the site-packages directory
set_target_properties(_judo_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${SITE_PACKAGES_DIR}/judo_cpp"
)  # rebuild the bound module to the site-packages directory

# Configure RPATHs correctly for each platform
if(APPLE)
  set_target_properties(_judo_cpp PROPERTIES
    MACOSX_RPATH ON
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "@loader_path/../mujoco"
    INSTALL_RPATH "@loader_path/../mujoco"
  )
  # MuJoCo pip wheel on macOS ships a flat lib, but exposes a framework-style install_name.
  # Rewrite the dependency on the built extension to the flat @rpath/libmujoco.*.dylib.
  get_filename_component(MUJOCO_LIB_NAME "${MUJOCO_LIB}" NAME)
  add_custom_command(TARGET _judo_cpp POST_BUILD
    COMMAND install_name_tool -change
      "@rpath/mujoco.framework/Versions/A/${MUJOCO_LIB_NAME}"
      "@rpath/${MUJOCO_LIB_NAME}"
      "$<TARGET_FILE:_judo_cpp>"
    VERBATIM)
else()
  set_target_properties(_judo_cpp PROPERTIES
    BUILD_RPATH "$ORIGIN/../mujoco"
    INSTALL_RPATH "$ORIGIN/../mujoco"
    BUILD_RPATH_USE_ORIGIN TRUE
  )
endif()
target_link_libraries(_judo_cpp PRIVATE
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  ${MUJOCO_LIB}
)

# 6. Link ONNX Runtime from the current environment (expects conda/pixi env)
if(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX $ENV{CONDA_PREFIX})
  include_directories("${CONDA_PREFIX}/include")
  file(GLOB ONNXRUNTIME_LIB_FILES
       "${CONDA_PREFIX}/lib/libonnxruntime.*"      # Linux/macOS
       "${CONDA_PREFIX}/Library/lib/onnxruntime.lib"  # Windows import lib
       "${CONDA_PREFIX}/Library/bin/onnxruntime.dll"  # Windows DLL
  )
  if(ONNXRUNTIME_LIB_FILES)
    list(GET ONNXRUNTIME_LIB_FILES 0 ONNXRUNTIME_LIB)
    message(STATUS "Linking against ONNX Runtime: ${ONNXRUNTIME_LIB}")
    target_link_libraries(_judo_cpp PRIVATE ${ONNXRUNTIME_LIB})
  else()
    message(WARNING "Could not find ONNX Runtime library under ${CONDA_PREFIX}. Ensure onnxruntime is installed.")
  endif()
else()
  message(WARNING "CONDA_PREFIX not set; skipping ONNX Runtime linking.")
endif()

# 5. Install the module
install(TARGETS _judo_cpp DESTINATION judo_cpp)
