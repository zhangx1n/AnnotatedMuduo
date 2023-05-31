# 定义参与编译的源代码文件
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
message(${PROJECT_SOURCE_DIR})
#AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR} SRC_LIST)    # 这里会检索子目录的文件,会包括example中的testserver
#设置子目录
set(SUB_DIR_LIST ${PROJECT_SOURCE_DIR}/mymuduo ${PROJECT_SOURCE_DIR}/example)

foreach(SUB_DIR ${SUB_DIR_LIST})
    #遍历源文件
    aux_source_directory(${SUB_DIR} SRC_LIST)
endforeach()

add_executable(testserver ${SRC_LIST})

TARGET_LINK_LIBRARIES(testserver pthread)