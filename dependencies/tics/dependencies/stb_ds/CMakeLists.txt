project(stb_ds)

add_library(${PROJECT_NAME} STATIC src/stb_ds.c)
# Prevents consumers from accidentally modifying compiler flags on this lib.
add_library(tics_stb_ds ALIAS stb_ds)
target_include_directories(${PROJECT_NAME} PUBLIC include)
