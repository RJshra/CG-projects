cmake_minimum_required(VERSION 3.10)

project(get_start)

set(THIRD_PARTY_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/external)

file(GLOB PROJECT_HDR ./*.h)
file(GLOB PROJECT_SRC ./*.cpp)
set(BASE_HDR ../base/application.h
             ../base/frame_rate_indicator.h
             ../base/input.h
             ../base/glsl_program.h
             ../base/camera.h
             ../base/frustum.h
             ../base/plane.h
             ../base/transform.h
             ../base/sampler.h
             ../base/texture.h
             ../base/texture2d.h
             ../base/bounding_box.h
             ../base/vertex.h)
set(BASE_SRC ../base/application.cpp 
             ../base/glsl_program.cpp 
             ../base/camera.cpp 
             ../base/texture.cpp
             ../base/texture2d.cpp
             ../base/transform.cpp
)

add_executable(get_start ${PROJECT_SRC} ${PROJECT_HDR} ${BASE_SRC} ${BASE_HDR})

if(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(XCODE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
else()
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
    else()
        set_target_properties(${PROJECT_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
    endif()
endif()

target_include_directories(get_start PUBLIC ${THIRD_PARTY_LIBRARY_PATH}/glm)
target_include_directories(get_start PUBLIC ${THIRD_PARTY_LIBRARY_PATH}/glad/include)
target_include_directories(get_start PUBLIC ${THIRD_PARTY_LIBRARY_PATH}/glfw/include)
target_include_directories(get_start PUBLIC ${THIRD_PARTY_LIBRARY_PATH}/imgui)
target_include_directories(get_start PUBLIC ${THIRD_PARTY_LIBRARY_PATH}/stb)
target_include_directories(get_start PUBLIC ${THIRD_PARTY_LIBRARY_PATH}/eigen-3.4.0)
target_link_libraries(get_start glm)
target_link_libraries(get_start glad)
target_link_libraries(get_start glfw)
target_link_libraries(get_start imgui)
target_link_libraries(get_start stb)