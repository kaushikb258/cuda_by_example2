#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void myfunc()
{
}

int main(void)
{
 myfunc<<<1,1>>>();
 cout<<"Hello World from host! \n";
 return 0;
}
