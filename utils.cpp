#include "utils.h"
READFILE_RESULT readfile(const string& _filename, DataSet& _infodata) {
    fstream fin;
    fin.open(_filename.c_str(), fstream::in);
    if (fin.fail()) {
        return CANNOT_READ_FILE;
    }
    // read file line by line
    string line;
    InfoData info;
    while (getline(fin, line)) {
        if (line.empty()) break;
        istringstream ss(line);
        ss >> info.first;
        ss >> info.second;
        _infodata.push_back(info);
    }
    fin.close();
    return SUCCESS;
}
