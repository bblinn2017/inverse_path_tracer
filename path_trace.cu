#include <stdio.h>
#include "scene_basics.h"

int main(int argc, char *argv[]) {
    float position[] = {0.,0.,0.};
    float orientation[] = {0.,0.,0.};
    float scale[] = {1.,1.,1.};
    shape_t shape = shape_t::Cube;
    std::string obj_file = "";
    std::string mtl_file = "";

    Object obj = Object(position,orientation,scale,shape,obj_file,mtl_file);
}