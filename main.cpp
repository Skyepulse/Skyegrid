#include <iostream>
#include "includes/SkyegridManager.hpp"

//================================//
int main()
{
    SkyegridManager manager;
    manager.RunMainLoop();

    std::cout << "Exiting application.\n";
    return 0;
}