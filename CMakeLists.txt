cmake_minimum_required(VERSION 3.15)
project(tower-stack)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(TARGET_NAME ${PROJECT_NAME})

# configure tics
set(TICS_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(TICS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TICS_ENABLE_DEBUG_VIEW OFF CACHE BOOL "" FORCE)
set(TICS_ENABLE_PROFILER OFF CACHE BOOL "" FORCE)
set(TICS_BUILD_TOOL_GLTF2C OFF CACHE BOOL "" FORCE)
set(TICS_ENABLE_CUDA OFF CACHE BOOL "" FORCE)

add_subdirectory(dependencies/tics)
add_subdirectory(dependencies/sokol_util)

add_executable(${TARGET_NAME} main.c)

target_include_directories(${TARGET_NAME} PRIVATE
	dependencies/tics/include
	dependencies/sokol
	dependencies/sokol_util
)

target_link_libraries(${TARGET_NAME} PRIVATE
	tics
	sokol_util
)

if(EMSCRIPTEN)
	# Tell Emscripten to use WebGL2 (required by Sokol GLES3 backend) and allow memory growth
	target_link_options(${TARGET_NAME} PRIVATE
		"-sMIN_WEBGL_VERSION=2"
		"-sMAX_WEBGL_VERSION=2"
		"-sALLOW_MEMORY_GROWTH=1"
		"--shell-file" "${CMAKE_CURRENT_SOURCE_DIR}/shell.html"
	)
	set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".html")
else()
	if(CMAKE_SYSTEM_NAME MATCHES "Linux")
		# Linux OpenGL & X11 dependencies
		target_link_libraries(${TARGET_NAME} PRIVATE X11 Xi Xcursor GL m)
	elseif(WIN32)
		# This project requires MinGW, which requires explicit D3D11 linking
		target_link_libraries(${TARGET_NAME} PRIVATE d3d11 dxgi)
	endif()
endif()
