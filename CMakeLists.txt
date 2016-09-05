cmake_minimum_required(VERSION 2.8)

PROJECT (Juno)

find_package(OpenCV  REQUIRED)


FOREACH( mesdossiers 
        JunoRaw
    )
    ADD_SUBDIRECTORY(${mesdossiers})

ENDFOREACH()




