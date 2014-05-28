#-------------------------------------------------
#
# Project created by QtCreator 2014-05-12T18:01:45
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = PlaceRecognition
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    BowVocabulary.cpp \
    DataParser.cpp \
    examples.cpp \
    datahandler.cpp \
    bayesExample.cpp \
    svmclassifier.cpp

HEADERS += \
    BowVocabulary.h \
    DataParser.h \
    engine.h \
    examples.h \
    tinydir.h \
    datahandler.h \
    svmclassifier.h

QMAKE_CXXFLAGS += -std=c++0x

INCLUDEPATH += /usr/local/include/opencv \
    /usr/include/boost
LIBS += -L/usr/local/lib \
    -lopencv_core \
    -lopencv_video \
    -lopencv_highgui \
    -lopencv_features2d \
    -lopencv_imgproc \
    -lopencv_ml \
    -lopencv_calib3d \
    -lopencv_objdetect \
    -lopencv_contrib \
    -lopencv_legacy \
    -lopencv_flann \
    -lopencv_nonfree \
    -lboost_system \
    -lboost_iostreams \
    -lboost_filesystem
