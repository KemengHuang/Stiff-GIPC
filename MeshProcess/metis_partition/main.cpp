#include <metis.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <filesystem>
#include "node_edge_model.h"
#include "mesh.h"
#include "metis_sort.h"

int main()
{
    std::string mesh_name = "high_cloth";
    std::string obj_path   = ASSETS_DIR "/triMesh/" + mesh_name + ".obj";
    metis_sort(obj_path, 2);
    return 0;
}