cmake_minimum_required	(VERSION 2.8)
project			(PlaceRecognitionCmake)

set			(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include_directories	(${CMAKE_CURRENT_SOURCE_DIR})

# Version number
set			(PlaceRecognitionCmake_VERSION_MAJOR 1)
set			(PlaceRecognitionCmake_VERSION_MINOR 1)

# For OpenCV
find_package		(OpenCV REQUIRED)
if			(OpenCV_FOUND)
list			(APPEND ThirdParty_LIBS ${OpenCV_LIBS})
	include_directories	(${OPenCV_INCLUDE_DIRS})
endif			(OpenCV_FOUND)

# For Boost
set			(Boost_USE_STATIC_LIBS OFF)
set			(Boost_USE_MULTITHREADED ON)
set			(Boost_USE_STATIC_RUNTIME OFF)

find_package		(Boost 1.45.0 COMPONENTS system iostreams filesystem)
if			(Boost_FOUND)
	include_directories	(${Boost_INCLUDE_DIRS})
endif			(Boost_FOUND)

# For C++11
set			(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11")

set			(${PROJECT_NAME}_SRC main.cpp BowVocabulary.cpp DataParser.cpp examples.cpp datahandler.cpp bayesExample.cpp svmclassifier.cpp foldsextraction.cpp fold.cpp crossvalidation.cpp)

set			(${PROJECT_NAME}_HDR BowVocabulary.h DataParser.h engine.h examples.h tinydir.h datahandler.h svmclassifier.h foldsextraction.h fold.h crossvalidation.h)

add_executable		(${PROJECT_NAME} ${${PROJECT_NAME}_SRC} ${${PROJECT_NAME}_HDR})

target_link_libraries	(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
# To link all boost libraries use next line instead
#target_link_libraries	(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_PROGRAM_OPTIONS_LIBRARY)} ${Boost_REGEX_LIBRARY})

