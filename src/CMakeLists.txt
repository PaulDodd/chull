set(_chull_sources  ) # No sources

set(_chull_headers
    chull.h
    )

# copy python modules to the build directory to make it a working python package
message(${CMAKE_CURRENT_BINARY_DIR})
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/include/chull/${file}
        COMMENT    "Copy include/chull/${file}"
    )
ENDMACRO(copy_header)

add_custom_target(copy_chull ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_chull_headers}
        DESTINATION include/chull
       )

# copy headers into build directory
foreach(file ${_chull_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_chull_headers ALL DEPENDS ${_pynomial_headers})
