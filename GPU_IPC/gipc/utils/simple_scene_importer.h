#pragma once
#include <load_mesh.h>
#include <nlohmann/json.hpp>
#include <string>
#include <gipc/body_type.h>

namespace gipc
{
class SimpleSceneImporter
{
    using json = nlohmann::json;

  public:
    SimpleSceneImporter() = default;
    SimpleSceneImporter(std::string_view scene_file_name,
                        std::string_view mesh_dir,
                        BodyType         body_type = BodyType::ABD);
    json        scene_json;
    std::string mesh_dir;
    BodyType    body_type;
    void        import_scene(tetrahedra_obj& tetras);
    void        load_geometry(tetrahedra_obj& tetras,
                              int             Dimensions,
                              gipc::BodyType  bodyType,
                              Eigen::Matrix4d transform,
                              double          YoungthM,
                              std::string     meth_path,
                              int             preconditionerType);
};

}  // namespace gipc
