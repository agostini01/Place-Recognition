#include "datahandler.h"

dataHandler::dataHandler()
{
}

void dataHandler::tinydirExample()
{
    tinydir_dir dir;
    unsigned int i;
    tinydir_open_sorted(&dir, "../data/test/bedroom");

    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        tinydir_readfile_n(&dir, &file, i);

        printf("%s", file.name);
        if (file.is_dir)
        {
            printf("/");
        }
        printf("\n");
    }

    tinydir_close(&dir);
}

std::list<std::string> dataHandler::filesIn(const char* folderName)
{
    std::list<std::string> filesList;
    tinydir_dir dir;
    unsigned int i;
    tinydir_open_sorted(&dir, folderName);

    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        tinydir_readfile_n(&dir, &file, i);
        if (!file.is_dir)
        {
            filesList.push_back(file.name);
        }
    }

    tinydir_close(&dir);
    return filesList;
}

std::list<std::string> dataHandler::foldersIN(const char *folderName)
{
    std::list<std::string> foldersList;
    tinydir_dir dir;
    unsigned int i;
    tinydir_open_sorted(&dir, folderName);

    //  starts from 2 to get rid of ./ and ../
    for (i = 2; i < dir.n_files; i++)
    {
        tinydir_file file;
        tinydir_readfile_n(&dir, &file, i);
        if (file.is_dir)
        {
            foldersList.push_back(file.name);
        }
    }

    tinydir_close(&dir);
    return foldersList;
}

std::map<std::string, std::list<std::string> > dataHandler::buildDataFiles(const char *folderName)
{
    std::map<std::string, std::list<std::string> > myMap;
    std::list<std::string> foldersList = foldersIN(folderName);
    std::string path;
    for (auto it1 = foldersList.begin(); it1 != foldersList.end(); ++it1) {
        path.clear();
        path.append(folderName);
        path.append((*it1));
        path.append("/");
        myMap[(*it1)]=filesIn((path.c_str()));
    }
    return myMap;
}
