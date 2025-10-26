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
std::string get_filename_without_extension(const std::string& path)
{
    std::filesystem::path p(path);
    return p.stem().string();  // stem() 直接获取不带扩展名的文件名
}

std::string get_extension(const std::string& path)
{
    std::filesystem::path p(path);
    return p.extension().string();  // extension() 直接获取扩展名（包含点号）
}

std::vector<std::string> metis_sort(std::string obj_path, int dimension)
{
    std::string mesh_name     = get_filename_without_extension(obj_path);
    std::string extension     = get_extension(obj_path);
    bool        isTriangle    = true;
    size_t      block_size    = 16;
    std::string output_folder = OUTPUT_DIR;
    std::string out_file_path = output_folder + mesh_name + "_sorted."
                                + std::to_string(block_size) + extension;

    std::filesystem::exists(output_folder)
        || std::filesystem::create_directory(output_folder);
    std::ifstream ifs(out_file_path);
    if(ifs)
    {
        printf("metis files exist");
        ifs.close();
        std::vector<std::string> out_paths;
        std::string sort_part_path = output_folder + mesh_name + "_sorted."
                                     + std::to_string(block_size) + ".part";

        std::string sort_obj_path = output_folder + mesh_name + "_sorted."
                                    + std::to_string(block_size) + extension;

        out_paths.push_back(out_file_path);
        out_paths.push_back(sort_part_path);
        return out_paths;
    }
    ifs.close();


    if(dimension == 3)
        isTriangle = false;

    if(isTriangle)
    {
        gipc::TriMesh mesh;
        mesh.load(obj_path);
        std::vector<idx_t> part;

        std::cout << "vertex num: " << mesh.vertices().size() << std::endl;
        //std::cout << "tetrahedra num: " << mesh.tetrahedra().size() << std::endl;
        std::cout << "block size: " << block_size << std::endl;
        std::cout << "partitioning..." << std::endl;

        auto nPart = (mesh.vertices().size() + block_size - 2) / (block_size - 1);
        // auto nPart = (mesh.vertices().size() + block_size) / (block_size);

        if(nPart <= 1)
        {
            for(int i = 0; i < mesh.vertices().size(); i++)
            {
                part.push_back(0);
            }
        }
        else
        {
            mesh.k_way_partition(nPart, part);
        }
        std::cout << "partitioning done" << std::endl;

        // statistics
        std::vector<std::vector<int>> blocks(nPart);
        for(int i = 0; i < part.size(); ++i)
        {
            blocks[part[i]].push_back(i);
            //std::cout << "part:  "<< part[i] << "    " << i << std::endl;
            //if(part[i] == 232)
            //{
            //    std::cout  << i << std::endl;
            //}
        }

        // average block size
        auto avg_block_size = std::accumulate(blocks.begin(),
                                              blocks.end(),
                                              0.0,
                                              [](int a, const auto& b) -> double
                                              { return a + b.size(); })
                              / blocks.size();

        auto min_block_size = std::min_element(blocks.begin(),
                                               blocks.end(),
                                               [](const auto& a, const auto& b)
                                               { return a.size() < b.size(); });

        auto max_block_size = std::max_element(blocks.begin(),
                                               blocks.end(),
                                               [](const auto& a, const auto& b)
                                               { return a.size() < b.size(); });

        // block that not full
        auto not_full_block = std::count_if(blocks.begin(),
                                            blocks.end(),
                                            [block_size](const auto& b)

                                            { return b.size() < block_size; });

        std::cout << "min block size: " << min_block_size->size() << std::endl;
        std::cout << "non-full-block count: " << not_full_block << std::endl;

        {
            for(int i = 0, j = 0; j < blocks.size(); j++)
            {
                if(blocks[j].size() < block_size)
                {
                    std::cout << "-- [" << i++ << "] id=" << j
                              << " size=" << blocks[j].size() << std::endl;
                }
            }
        }

        std::cout << std::endl;


        std::cout << "total block count: " << blocks.size() << std::endl;
        std::cout << "average block size: " << avg_block_size << std::endl;
        std::cout << "max block size: " << max_block_size->size() << std::endl;

        auto exceed_block = std::count_if(blocks.begin(),
                                          blocks.end(),
                                          [block_size](const auto& b)
                                          { return b.size() > block_size; });

        std::cout << "exceed-block count: " << exceed_block << std::endl;

        {
            for(int i = 0, j = 0; j < blocks.size(); j++)
            {
                if(blocks[j].size() > block_size)
                {
                    std::cout << "-- [" << i++ << "] id=" << j
                              << " size=" << blocks[j].size() << std::endl;
                }
            }
        }

        std::cout << "calculate sort index..." << std::endl;
        auto sort_index = mesh.sort_index(part);


        //std::cout << "export wireframe..." << std::endl;
        //
        //std::filesystem::exists(output_folder)
        //    || std::filesystem::create_directory(output_folder);
        //mesh.export_wireframe(output_folder + mesh_name + "."
        //                      + std::to_string(block_size) + ".obj");

        //std::cout << "export partition..." << std::endl;
        //mesh.export_partition(output_folder + mesh_name + "."
        //                          + std::to_string(block_size) + ".part",
        //                      part);


        //std::cout << "export sorted index ..." << std::endl;
        //mesh.export_sort_index(output_folder + mesh_name + "."
        //                           + std::to_string(block_size) + ".sort_index",
        //                       part);

        auto sorted_mesh = mesh.sorted(sort_index);
        //std::cout << "export sorted wireframe ..." << std::endl;

        std::vector<int> sorted_part(part.size());
        for(int i = 0; i < part.size(); i++)
        {
            sorted_part[i] = part[sort_index[i]];
        }
        //sorted_mesh.export_wireframe(output_folder + mesh_name + "_sorted."
        //                             + std::to_string(block_size) + ".obj");
        std::vector<std::string> out_paths;
        std::string sort_part_path = output_folder + mesh_name + "_sorted."
                                     + std::to_string(block_size) + ".part";
        std::cout << "export sorted partition ..." << std::endl;
        sorted_mesh.export_partition(output_folder + mesh_name + "_sorted."
                                         + std::to_string(block_size) + ".part",
                                     sorted_part);
        std::string sort_obj_path = output_folder + mesh_name + "_sorted."
                                    + std::to_string(block_size) + extension;
        std::cout << "export sorted tri mesh ..." << std::endl;
        sorted_mesh.export_mesh(output_folder + mesh_name + "_sorted."
                                + std::to_string(block_size) + extension);

        out_paths.push_back(sort_obj_path);
        out_paths.push_back(sort_part_path);
        return out_paths;
    }
    else
    {
        gipc::TetMesh mesh;
        mesh.load(obj_path);
        std::vector<idx_t> part;
        //size_t             block_size = 32;
        std::cout << "vertex num: " << mesh.vertices().size() << std::endl;
        //std::cout << "tetrahedra num: " << mesh.tetrahedra().size() << std::endl;
        std::cout << "block size: " << block_size << std::endl;
        std::cout << "partitioning..." << std::endl;

        auto nPart = (mesh.vertices().size() + block_size - 2) / (block_size - 1);
        // auto nPart = (mesh.vertices().size() + block_size) / (block_size);

        if(nPart <= 1)
        {
            for(int i = 0; i < mesh.vertices().size(); i++)
            {
                part.push_back(0);
            }
        }
        else
        {
            mesh.k_way_partition(nPart, part);
        }

        std::cout << "partitioning done" << std::endl;

        // statistics
        std::vector<std::vector<int>> blocks(nPart);
        for(int i = 0; i < part.size(); ++i)
        {
            blocks[part[i]].push_back(i);
            //std::cout << "part:  "<< part[i] << "    " << i << std::endl;
            //if(part[i] == 232)
            //{
            //    std::cout  << i << std::endl;
            //}
        }

        // average block size
        auto avg_block_size = std::accumulate(blocks.begin(),
                                              blocks.end(),
                                              0.0,
                                              [](int a, const auto& b) -> double
                                              { return a + b.size(); })
                              / blocks.size();

        auto min_block_size = std::min_element(blocks.begin(),
                                               blocks.end(),
                                               [](const auto& a, const auto& b)
                                               { return a.size() < b.size(); });

        auto max_block_size = std::max_element(blocks.begin(),
                                               blocks.end(),
                                               [](const auto& a, const auto& b)
                                               { return a.size() < b.size(); });

        // block that not full
        auto not_full_block = std::count_if(blocks.begin(),
                                            blocks.end(),
                                            [block_size](const auto& b)

                                            { return b.size() < block_size; });

        std::cout << "min block size: " << min_block_size->size() << std::endl;
        std::cout << "non-full-block count: " << not_full_block << std::endl;

        {
            for(int i = 0, j = 0; j < blocks.size(); j++)
            {
                if(blocks[j].size() < block_size)
                {
                    std::cout << "-- [" << i++ << "] id=" << j
                              << " size=" << blocks[j].size() << std::endl;
                }
            }
        }

        std::cout << std::endl;


        std::cout << "total block count: " << blocks.size() << std::endl;
        std::cout << "average block size: " << avg_block_size << std::endl;
        std::cout << "max block size: " << max_block_size->size() << std::endl;

        auto exceed_block = std::count_if(blocks.begin(),
                                          blocks.end(),
                                          [block_size](const auto& b)
                                          { return b.size() > block_size; });

        std::cout << "exceed-block count: " << exceed_block << std::endl;

        {
            for(int i = 0, j = 0; j < blocks.size(); j++)
            {
                if(blocks[j].size() > block_size)
                {
                    std::cout << "-- [" << i++ << "] id=" << j
                              << " size=" << blocks[j].size() << std::endl;
                }
            }
        }

        std::cout << "calculate sort index..." << std::endl;
        auto sort_index = mesh.sort_index(part);


        /*std::cout << "export wireframe..." << std::endl;
        std::string output_folder = OUTPUT_DIR "/mesh_process/";
        std::filesystem::exists(output_folder)
            || std::filesystem::create_directory(output_folder);
        mesh.export_wireframe(output_folder + mesh_name + "."
                              + std::to_string(block_size) + ".obj");*/

        //std::cout << "export partition..." << std::endl;
        //mesh.export_partition(output_folder + mesh_name + "."
        //                          + std::to_string(block_size) + ".part",
        //                      part);


        //std::cout << "export sorted index ..." << std::endl;
        //mesh.export_sort_index(output_folder + mesh_name + "."
        //                           + std::to_string(block_size) + ".sort_index",
        //                       part);

        auto sorted_mesh = mesh.sorted(sort_index);
        //std::cout << "export sorted wireframe ..." << std::endl;

        std::vector<int> sorted_part(part.size());
        for(int i = 0; i < part.size(); i++)
        {
            sorted_part[i] = part[sort_index[i]];
        }
        //sorted_mesh.export_wireframe(output_folder + mesh_name + "_sorted."
        //                             + std::to_string(block_size) + ".obj");
        std::vector<std::string> out_paths;
        std::string sort_part_path = output_folder + mesh_name + "_sorted."
                                     + std::to_string(block_size) + ".part";
        std::cout << "export sorted partition ..." << std::endl;
        sorted_mesh.export_partition(output_folder + mesh_name + "_sorted."
                                         + std::to_string(block_size) + ".part",
                                     sorted_part);
        std::string sort_obj_path = output_folder + mesh_name + "_sorted."
                                    + std::to_string(block_size) + extension;
        std::cout << "export sorted tet mesh ..." << std::endl;
        sorted_mesh.export_mesh(output_folder + mesh_name + "_sorted."
                                + std::to_string(block_size) + extension);

        out_paths.push_back(sort_obj_path);
        out_paths.push_back(sort_part_path);
        return out_paths;
    }
}