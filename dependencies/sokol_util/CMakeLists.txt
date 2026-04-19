add_library(sokol_util STATIC sokol_util.c)
target_include_directories(sokol_util PUBLIC .)
# Sokol app header needs to be available
target_include_directories(sokol_util PRIVATE ../sokol)
