
cmake_minimum_required(VERSION 3.12)
project(mini_kgraph)



# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS system program_options)


include_directories(src ${Boost_INCLUDE_DIRS})


# Add executable
add_executable(test tests/test.cpp src/kgraph.cpp)

# Link libraries
target_link_libraries(test ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS} OpenMP::OpenMP_CXX)

