#include <iostream>
#include <fstream>
#include <string>

std::string* getLines(const std::string& fileName, size_t& arrSize){
    std::ifstream bundleFile;
    bundleFile.open(fileName);
    std::string line;

    size_t maxSize = 5;
    size_t currSize = 0;
    auto* lines = new std::string[maxSize];

    while(std::getline(bundleFile, line)){
        if(currSize == maxSize){
            // need to make lines bigger
            auto* newLines = new std::string[maxSize*2];
            for(size_t i = 0; i < maxSize; i++){
                newLines[i] = lines[i];
            }

            // no longer need lines, so clean up memory
            delete[] lines;
            lines = newLines;

            // and record new max size
            maxSize *= 2;
        }

        lines[currSize] = line;
        currSize++;
    }

    bundleFile.close();

    // return
    arrSize = currSize;
    return lines;
}

/**
 * Gets data about either a bundle or a series from a line.
 * @param line The line to read bundle information about
 * @param name (Return) The name of the bundle or series
 * @return Data about the bundle or series, terminated by -1. (Note that size_t is unsigned.)
 * Data format is designed in the PyCharm processor, but assuming nothing has changed since this was written:
 * Bundle Data Format:
 *  First item is bundle size
 *  Remaining items are the IDs of series in the bundle.
 * Series Data Format:
 *  First item is the series size
 *  Second item is the sizes of the series.
 *  Third item is -1
 */
size_t* getLineData(std::string line, std::string& name){
    name = "";
    size_t maxSize = 5;
    size_t currSize = 0;
    auto* arr = new size_t[maxSize];
    while(!line.empty()){
        auto index = line.find('$');
        std::string token;
        if(index != std::string::npos){
            token = line.substr(0, index);
            line = line.substr(index+1); // drop the $
        }
        else{
            token = line;
            line = "";
        }

        if(name.empty()){
            // token is name
            name = token;
        }
        else{
            // token is an item
            // check if we can fit the item:
            if(currSize == maxSize){
                // need to resize array for new item
                auto* newArr = new size_t[maxSize * 2];
                for(size_t i = 0; i < currSize; i++){
                    newArr[i] = arr[i];
                }

                // then delete the existing array
                delete[] arr;

                // then put the new array info in
                arr = newArr;
                maxSize *= 2;
            }

            // add item
            arr[currSize] = std::stoi(token);

            // item added, update currSize
            currSize++;
        }
    }
    // final array is prepared
    // trim it so it's not massive in memory
    auto* out = new size_t[currSize+1]; // because currSize item will be -1
    // copy over items
    for(size_t i = 0; i < currSize; i++){
        out[i] = arr[i];
    }
    // put the -1 item in
    out[currSize] = -1;
    // we don't need the old version so remove it to save memory
    delete[] arr;
    // and return
    return out;
}

/**
 * Int-to-string. (I-to-S)
 * Assumes str is large enough, and just overrides str.
 */
__device__ void deviceItos(char* &str, size_t num){
    if(num == 0){
        str[0] = '0';
        str[1] = NULL;
        return;
    }
    // find the power of 10 bigger than this
    size_t divBy = 1;
    while((num/divBy) != 0){
        divBy *= 10;
    }
    // num/divBy = 0, fix that:
    size_t strIdx = 0;
    while(divBy != 1){
        // remember at first num/divBy = 0 so don't print that 0
        divBy /= 10;
        str[strIdx] = '0' + ((num/divBy)%10); // NOLINT(cppcoreguidelines-narrowing-conversions)
        strIdx++;
    }
    str[strIdx] = NULL;
}

__device__ void deviceStrCat(char* dest, const char* src){
    size_t destIdx = 0;
    while(dest[destIdx]){
        destIdx++;
    }
    size_t srcIdx = 0;
    while(src[srcIdx]){
        dest[destIdx] = src[srcIdx];
        destIdx++;
        srcIdx++;
    }
    dest[destIdx] = NULL;
}