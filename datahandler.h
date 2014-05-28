#ifndef DATAHANDLER_H
#define DATAHANDLER_H

#include "tinydir.h"
#include <list>
#include <map>
#include <string>
#include <iostream>
#include <stdio.h>

class dataHandler
{
public:
    dataHandler();

    void tinydirExample();
    std::list<std::string> filesIn(const char* folderName);
    std::list<std::string> foldersIN(const char *folderName);
    std::map<std::string, std::list<std::string> > buildDataFiles(const char *folderName);

};

#endif // DATAHANDLER_H
