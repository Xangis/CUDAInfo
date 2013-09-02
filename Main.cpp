#include "CUDAInfo.h"
#include <iostream>

using namespace std;

int main( int argc, char *argv[])
{
    CUDAInfo* info = new CUDAInfo();
#ifdef WIN32
    cout << endl << "Done. Hit enter to close the program." << endl;
    char c;
    cin.get(c);
#endif
}
