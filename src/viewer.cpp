#include "rhoban_fa/tools/viewer.h"

#include <iostream>

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <faFile> <configFile>" << std::endl;
    exit(EXIT_FAILURE);
  }

  rhoban_fa::Viewer viewer(argv[1], argv[2], 1920, 1080);

  while (viewer.update())
  {
  }
}
