add_executable(patch)

set(SOURCE_LIST
    vertex_normal.cu 
    vertex_normal_ref.h
    vertex_normal_kernel.cuh
	vertex_normal_hardwired.cuh
)

target_sources(patch 
    PRIVATE
    ${SOURCE_LIST}
)

set_target_properties(patch PROPERTIES FOLDER "apps")

set_property(TARGET patch PROPERTY CUDA_SEPARABLE_COMPILATION ON)

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "patch" FILES ${SOURCE_LIST})

target_link_libraries( patch 
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)

#gtest_discover_tests( patch )