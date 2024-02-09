#include <Python.h>
#include </usr/include/python3.9/pyconfig-64.h>
#include <stdio.h>
int main()
{
    Py_Initialize();
    PyRun_SimpleString("print('Hello, World!')");
    Py_Finalize();
    return 0;
}