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
__device__ void deviceItos(char* &str, size_t num, const size_t base = 10, size_t minLen = 0){
    if(num == 0){
        str[0] = '0';
        str[1] = NULL;
        return;
    }
    // find the power of 10 bigger than this
    size_t compare = num;
    size_t strIdx = 0;
    while(compare > 0){
        // move over one digit
        // and move str over by one to accomodate
        compare /= base;
        strIdx++;
    }

    // str is now one past the last index
    // ie the null index
    strIdx--;

    if(strIdx < minLen){
        strIdx = minLen;
    }

    // Str is now at the very last letter
    // So right after this last letter should be a null
    str[strIdx+1] = '\0';

    while(num > 0){
        str[strIdx] = ('0' + (num%base)); // NOLINT(cppcoreguidelines-narrowing-conversions)
        strIdx--;
        num /= base;
    }

    // note that if strIdx == 0 then it needs to write to strIdx = 0
    // but if strIdx = -1 it has written to strIdx = 0
    while(strIdx != (-1)){
        // pad the left side with 0s
        // great for binary comparing
        str[strIdx] = '0';
        strIdx--;
    }
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
    dest[destIdx] = '\0';
}

/**
 * Prints a str and a num.
 * Effectively equivalent to converting the num to a str, concatenating the two, etc. etc.
 * but this does it all in one function. (This also appends a \\n afterwards, conveniently.)
 *
 * This function was created largely for the in-code Profiler to use.
 */
__device__ void devicePrintStrNum(const char* str, size_t num, size_t base = 10, size_t minLen = 0, bool newline = true){
    // add 2 just in case i messed something up somehow
    // because i'm pretty sure I messed something up and this is easier than checking
    size_t strlen = 2;
    const char* strptr = str;
    while(*strptr != NULL){
        strptr++;
        strlen++;
    }

    char* prntStr = new char[strlen+70];
    prntStr[0] = '\0';

    // Increased to 70 because the base-2 would need 64 chars at worst
    char* numStr = new char[70];

    deviceStrCat(prntStr, "[");

    deviceItos(numStr, blockIdx.x, 10, 2);
    deviceStrCat(prntStr, numStr);

    deviceStrCat(prntStr, ":");

    deviceItos(numStr, threadIdx.x, 10, 4);
    deviceStrCat(prntStr, numStr);

    deviceStrCat(prntStr, "] ");

    deviceStrCat(prntStr, str);

    deviceItos(numStr, num, base, minLen);
    deviceStrCat(prntStr, numStr);

    if(newline){
        printf("%s\n", prntStr);
    }
    else{
        printf("%s", prntStr);
    }

    delete[] numStr;
    delete[] prntStr;

}