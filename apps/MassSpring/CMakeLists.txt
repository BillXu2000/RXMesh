add_executable(MassSpring)

set(SOURCE_LIST
    mass_spring.cu 
    mass_spring_kernel.cuh
)

target_sources(MassSpring 
    PRIVATE
    ${SOURCE_LIST}
)

include_directories(
    "/usr/local/cuda/samples/common/inc"
)

set_target_properties(MassSpring PROPERTIES FOLDER "apps")

source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "MassSpring" FILES ${SOURCE_LIST})

target_link_libraries( MassSpring 
    PRIVATE RXMesh_header_lib 
    PRIVATE RXMesh_lib
    PRIVATE gtest_main
)

#gtest_discover_tests( VertexNormal )
