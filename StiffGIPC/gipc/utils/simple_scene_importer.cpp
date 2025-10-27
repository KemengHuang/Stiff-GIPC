#include <gipc/utils/simple_scene_importer.h>
#include <fstream>
#include <iostream>
#include <Eigen/Geometry>
#include <fem_parameters.h>
#include <metis_sort.h>

namespace gipc
{
template <typename T, int dim, int max_dim = dim>
void from_json(const nlohmann::json&                                  json,
               Eigen::Matrix<T, dim, 1, Eigen::ColMajor, max_dim, 1>& vector)
{
    typedef Eigen::Matrix<T, dim, 1, Eigen::ColMajor, max_dim, 1> Vector;
    std::vector<T> list = json.get<std::vector<T>>();
    vector              = Eigen::Map<Vector>(list.data(), long(list.size()));
}

Eigen::Vector3d read_angular_field(const nlohmann::json& field)
{
    int angular_dim = 3;

    Eigen::Vector3d v;
    if(field.is_number())
    {
        v.setZero();
        v[0] = field.get<double>();
    }
    else
    {
        assert(field.is_array());
        from_json(field, v);
    }

    // Convert to radians for easy use later
    v *= FEM::PI / 180.0;

    return v;
}

SimpleSceneImporter::SimpleSceneImporter(std::string_view scene_file_name,
                                         std::string_view mesh_dir,
                                         BodyType         body_type)
    : mesh_dir{mesh_dir}
    , body_type{body_type}
{
    scene_json = json::parse(std::ifstream(std::string{scene_file_name}));
}


void SimpleSceneImporter::load_geometry(tetrahedra_obj&  tetras,
                                        int              Dimensions,
                                        gipc::BodyType   bodyType,
                                        Eigen::Matrix4d  transform,
                                        double           YoungthM,
                                        std::string      meth_path,
                                        int              preconditionerType,
                                        BodyBoundaryType body_boundary_type)
{
    if(Dimensions == 3)
    {

        if(bodyType == gipc::BodyType::ABD)
        {
            tetras.load_tetrahedraMesh(meth_path, transform, 1e8, bodyType, body_boundary_type);
        }
        else if(bodyType == gipc::BodyType::FEM)
        {
            if(preconditionerType)
            {
                auto paths = metis_sort(meth_path, Dimensions);
                tetras.load_tetrahedraMesh(paths[0], transform, YoungthM, bodyType, body_boundary_type);
                tetras.load_parts(paths[1]);
            }
            else
            {
                tetras.load_tetrahedraMesh(meth_path, transform, YoungthM, bodyType, body_boundary_type);
            }
        }
    }
    else if(Dimensions == 2)
    {
        if(preconditionerType)
        {
            auto paths = metis_sort(meth_path, Dimensions);
            tetras.load_triMesh(paths[0], transform, 0);
            tetras.load_parts(paths[1]);
        }
        else
        {
            tetras.load_triMesh(meth_path, transform, 0);
        }
    }
}


void SimpleSceneImporter::import_scene(tetrahedra_obj& tetras)
{
    const auto& rigid_bodies = scene_json["rigid_body_problem"]["rigid_bodies"];
    for(auto& rigid_body : rigid_bodies)
    {
        const auto& mesh = rigid_body["mesh"];

        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d R         = Eigen::Matrix3d::Identity();

        auto T = Eigen::Transform<double, 4, Eigen::Affine>::Identity();

        if(rigid_body.find("rotation") != rigid_body.end())
        {
            Eigen::Vector3d rotation = read_angular_field(rigid_body["rotation"]);
            R = (Eigen::AngleAxisd(rotation.z(), Eigen::Vector3d::UnitZ())
                 * Eigen::AngleAxisd(rotation.y(), Eigen::Vector3d::UnitY())
                 * Eigen::AngleAxisd(rotation.x(), Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
        }

        Eigen::Vector3d translation;
        from_json(rigid_body["position"], translation);
        translation -= Eigen::Vector3d(11., 0., 30.);
        transform.block<3, 3>(0, 0) = R;
        transform.block<3, 1>(0, 3) = translation;
        //transform *= Eigen::Vector4d{0.1, 0.1, 0.1, 1.0}.asDiagonal();

        auto is_fixed = rigid_body["is_dof_fixed"].get<bool>();

        auto mesh_file = mesh_dir + std::string{mesh};
        // std::cout << "mesh_file: " << mesh_file << std::endl;
        auto boundary_type = is_fixed ? BodyBoundaryType::Fixed : BodyBoundaryType::Free;
        //tetras.load_tetrahedraMesh(mesh_file, transform, 1e8, body_type, boundary_type);

        load_geometry(tetras, 3, body_type, transform, 1e8, mesh_file, 1, boundary_type);
    }
}

}  // namespace gipc
