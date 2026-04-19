cmake_minimum_required(VERSION 3.15)
project(tics)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON) # We need gnu11 for stb_ds.h

if(MSVC)
	message(FATAL_ERROR 
		"MSVC (Visual Studio) is not supported. "
		"This project relies on GNU C extensions (gnu11) for stb_ds.h and OpenMP features "
		"that MSVC's C compiler does not support. "
		"Please build using a MinGW-w64 toolchain (GCC or Clang) instead."
	)
endif()

# --- Define Build Options ---

# Options to enable or disable demos, tests, tools,...
# By default everything will be built, but can be disabled like this in a CMakeLists.txt:
#   set(TICS_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
# or when running cmake:
#   cmake -DTICS_BUILD_DEMOS=OFF

option(TICS_BUILD_DEMOS "Build demo applications" ON)
option(TICS_BUILD_TESTS "Build tests" ON)
option(TICS_ENABLE_DEBUG_VIEW "Enable Blick, a visual debugger (unix only)" ON)
option(TICS_ENABLE_PROFILER "Print profiling messages to stdout" ON)
option(TICS_BUILD_TOOL_GLTF2C "Build tool gltf2c (converts glb model to C source code)" OFF)
option(TICS_ENABLE_CUDA "Enable CUDA broad phase" OFF)

if(TICS_ENABLE_DEBUG_VIEW)
	if(NOT UNIX)
		message(WARNING 
			"TICS_ENABLE_DEBUG_VIEW was requested, but the current platform is not Unix-like.\n"
			"Blick relies on POSIX Shared Memory and Forking.\n"
			"--> Automatically disabling TICS_ENABLE_DEBUG_VIEW."
		)
		set(TICS_ENABLE_DEBUG_VIEW OFF CACHE BOOL "" FORCE)
	endif()
endif()

if(TICS_BUILD_TESTS)
	if(NOT UNIX)
		message(WARNING 
			"TICS_BUILD_TESTS was requested, but the current platform is not Unix-like.\n"
			"Testing relies on POSIX functionality.\n"
			"--> Automatically disabling TICS_BUILD_TESTS."
		)
		set(TICS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	endif()
endif()

# Enable LTO/IPO only for Release configuration
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- The Tics library ---

# Dependency: stb_ds.h (C library for dynamic arrays and hash tables)
add_subdirectory(dependencies/stb_ds)

set(SOURCES
	src/core.c
	src/step.c
	src/dynamics.c
	src/collision_broad_phase.c
	src/collision_narrow_phase.c
	src/gpu_broad_phase_adapter.c
)
add_library(${PROJECT_NAME} ${SOURCES})

if(TICS_ENABLE_CUDA)
	# Check for CUDA support
	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		add_subdirectory(dependencies/broad_phase_cuda)
		target_link_libraries(${PROJECT_NAME} PRIVATE broad_phase_cuda)
		target_compile_definitions(${PROJECT_NAME} PRIVATE TICS_HAS_GPU_BROAD_PHASE)

		# Strategy selection
		set(GPU_BROADPHASE_STRATEGY "B_HALF_SHELL" CACHE STRING "Select the GPU broadphase strategy")
		set_property(CACHE GPU_BROADPHASE_STRATEGY PROPERTY STRINGS "A" "B_HALF_SHELL" "B_NAIVE")
		# Expose the selection to the C via preprocessor defines
		if(GPU_BROADPHASE_STRATEGY STREQUAL "A")
			add_compile_definitions(GPU_STRATEGY_A)
			message(STATUS "GPU Broadphase Strategy: A")
		elseif(GPU_BROADPHASE_STRATEGY STREQUAL "B_HALF_SHELL")
			add_compile_definitions(GPU_STRATEGY_B_HALF_SHELL)
			message(STATUS "GPU Broadphase Strategy: B Half Shell")
		elseif(GPU_BROADPHASE_STRATEGY STREQUAL "B_NAIVE")
			add_compile_definitions(GPU_STRATEGY_B_NAIVE)
			message(STATUS "GPU Broadphase Strategy: B Naive")
		else()
			message(FATAL_ERROR "Unknown GPU_BROADPHASE_STRATEGY selected: ${GPU_BROADPHASE_STRATEGY}")
		endif()

	else()
		message(WARNING "CUDA compiler not found — disabling TICS_ENABLE_CUDA")
		set(TICS_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
	endif()
endif()

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME} PRIVATE tics_stb_ds)
target_link_libraries(${PROJECT_NAME} PRIVATE m) # math system library

if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_C_COMPILER_ID MATCHES "Clang")
	# link openmp statically because linux distros don't ship the clang version
	# also the OS might select an incompatible version
	# (eg fedora 42 will compile with clang 20 but use /usr/lib64/llvm18/lib/libomp.so when running)
	# libomp.a was compiled on Ubuntu 22.04 with llvmorg-20.1.8 to hopefully be somewhat compatible
	target_link_libraries(${PROJECT_NAME} PRIVATE
		"${CMAKE_SOURCE_DIR}/dependencies/omp/libomp.a"
		pthread # required for openmp
		dl # required for openmp
	)
	# also include headers so we dont need libomp development packages
	target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/dependencies/omp")
else()
	# Windows (MinGW), macOS, and Linux GCC
	# Everything should just work with the system libraries. Not entirely sure about macOS
	target_link_libraries(${PROJECT_NAME} PRIVATE -fopenmp)
endif()

if(NOT EMSCRIPTEN)
	# TODO Fix
	# Temporary fix to always use SIMD
	target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp -march=native)	
	# TODO Fix
	# This is bad for portability, but for now it's fine
	target_compile_options(${PROJECT_NAME} PRIVATE
		$<$<AND:$<CONFIG:Release>,$<C_COMPILER_ID:GNU,Clang>>:-march=native>
	)
endif()

if(TICS_ENABLE_PROFILER)
	target_compile_definitions(${PROJECT_NAME} PUBLIC TICS_ENABLE_PROFILER)
endif()

# We put all executables in the same folder (build/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
