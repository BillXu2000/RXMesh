#include <cuda_profiler_api.h>
#include "gtest/gtest.h"
#include "rxmesh/rxmesh_attribute.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/util/report.h"
#include "rxmesh/util/timer.h"
#include "mass_spring_kernel.cuh"
#include <iostream>

struct arg
{
    std::string obj_file_name = STRINGIFY(INPUT_DIR) "sphere3.obj";
    std::string output_folder = STRINGIFY(OUTPUT_DIR);
    uint32_t    num_run = 1;
    uint32_t    device_id = 0;
    char**      argv;
    int         argc;
    bool        shuffle = false;
    bool        sort = false;
    bool ev = false;
    bool vv = false;
} Arg;

template <typename T, uint32_t patchSize>
void mass_spring_rxmesh(RXMESH::RXMeshStatic<patchSize>&          rxmesh_static,
                        const std::vector<std::vector<T>>&        Verts,
                        const std::vector<std::vector<uint32_t>>& Faces)
{
    using namespace RXMESH;
    constexpr uint32_t blockThreads = 256;

    // Report
    Report report("MassSpring_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.obj_file_name, rxmesh_static);
    report.add_member("method", std::string("RXMesh"));
    std::string order = "default";
    if (Arg.shuffle) {
        order = "shuffle";
    } else if (Arg.sort) {
        order = "sorted";
    }
    report.add_member("input_order", order);
    report.add_member("blockThreads", blockThreads);

    RXMeshAttribute<T> ox, x;
    ox.set_name("ox");
	x.set_name("x");
    ox.init(Verts.size(), 3u, RXMESH::LOCATION_ALL);
	x.init(Verts.size(), 3u, RXMESH::LOCATION_ALL);
    // fill in the coordinates
    float mi[3], ma[3];
    for (uint32_t i = 0; i < Verts.size(); ++i) {
        for (uint32_t j = 0; j < Verts[i].size(); ++j) {
            if (i == 0) {
                mi[j] = ma[j] = Verts[i][j];
            }
            mi[j] = min(mi[j], Verts[i][j]);
            ma[j] = max(ma[j], Verts[i][j]);
        }
    }
    for (uint32_t i = 0; i < Verts.size(); ++i) {
        for (uint32_t j = 0; j < Verts[i].size(); ++j) {
            ox(i, j) = x(i, j) = (Verts[i][j] - mi[j]) / (ma[1] - mi[1]);
        }
    }
    auto print_momentum = [&]() {
        float sum[4];
        memset(sum, 0, sizeof(sum));
        for (int i = 0; i < Verts.size(); i++) {
            for (int j = 0; j < Verts[i].size(); j++) {
                float product = 1;
                for (int k = 0; k < 4; k++) {
                    sum[k] += product;
                    product *= x(i, j);
                }
            }
        }
        std::cerr << "momentum:\t";
        for (int i = 0; i < 4; i++) {
            std::cerr << sum[i] << "\t";
        }
        std::cerr << "\n";
    };
    auto print_obj = [&](std::string path) {
        std::fstream out(path, std::fstream::out);
        for (int i = 0; i < Verts.size(); i++) {
            out << "v " << x(i, 0) << " " << x(i, 1) << " " << x(i, 2) << "\n";
        }
        for (auto f : Faces) {
            out << "f " << f[0] + 1 << " " << f[1] + 1 << " " << f[2] + 1 << "\n";
        }
    };
    print_momentum();
    // move the coordinates to device
    ox.move(RXMESH::HOST, RXMESH::DEVICE);
		x.move(RXMESH::HOST, RXMESH::DEVICE);

    // velocity
    RXMeshAttribute<T> v;
    v.set_name("v");
    v.init(Verts.size(), 3u, RXMESH::LOCATION_ALL);

    // launch box
    LaunchBox<blockThreads> launch_box;
    rxmesh_static.prepare_launch_box(RXMESH::Op::VV, launch_box);

    LaunchBox<blockThreads> ev_launch_box;
    rxmesh_static.prepare_launch_box(RXMESH::Op::EV, ev_launch_box);


    TestData td;
    td.test_name = "MassSpring";

    float mass = 1.0 / Verts.size();
    std::cerr << mass << "\n";
    for (uint32_t itr = 0; itr < Arg.num_run; ++itr) {
        for (int j = 0; j < 100; j++) {
            if (Arg.vv)
                compute_mass_spring<T, blockThreads>
                    <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                        rxmesh_static.get_context(), ox, x, v, mass);
            if (Arg.ev) 
                ev_mass_spring<T, blockThreads>
                    <<<ev_launch_box.blocks, blockThreads, ev_launch_box.smem_bytes_dyn>>>(
                        rxmesh_static.get_context(), ox, x, v, mass);
            advect<T, blockThreads>
                <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                    rxmesh_static.get_context(), ox, x, v, mass);
        }
            
        /*advance<T, blockThreads>
            <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                rxmesh_static.get_context(), ox, x, v);*/
        /*x.move(RXMESH::DEVICE, RXMESH::HOST);
        print_momentum();
        print_obj("./results/" + std::to_string(itr) + ".obj");*/
    }
    x.move(RXMESH::DEVICE, RXMESH::HOST);
    print_momentum();


 
    // Release allocation
    ox.release();
    x.release();
		v.release();

    // Finalize report
    report.add_test(td);
    report.write(Arg.output_folder + "/rxmesh",
                 "MassSpring_RXMesh_" + extract_file_name(Arg.obj_file_name));
}

TEST(Apps, MassSpring)
{
    using namespace RXMESH;
    using dataT = float;

    if (Arg.shuffle) {
        ASSERT_FALSE(Arg.sort) << " cannot shuffle and sort at the same time!";
    }
    if (Arg.sort) {
        ASSERT_FALSE(Arg.shuffle)
            << " cannot shuffle and sort at the same time!";
    }

    // Select device
    cuda_query(Arg.device_id);

    // Load mesh
    std::vector<std::vector<dataT>>    Verts;
    std::vector<std::vector<uint32_t>> Faces;

    if (!import_obj(Arg.obj_file_name, Verts, Faces)) {
        exit(EXIT_FAILURE);
    }

    if (Arg.shuffle) {
        shuffle_obj(Faces, Verts);
    }

    // Create RXMeshStatic instance. If Arg.sort is true, Faces and Verts will
    // be sorted based on the patching happening inside RXMesh
    RXMeshStatic rxmesh_static(Faces, Verts, Arg.sort, false);

    //*** RXMesh Impl
    mass_spring_rxmesh(rxmesh_static, Verts, Faces);
}

int main(int argc, char** argv)
{
    using namespace RXMESH;
    Log::init();

    ::testing::InitGoogleTest(&argc, argv);
    Arg.argv = argv;
    Arg.argc = argc;

    if (argc > 1) {
        if (cmd_option_exists(argv, argc + argv, "-h")) {
            // clang-format off
            RXMESH_INFO("\nUsage: MassSpring.exe < -option X>\n"
                        " -h:          Display this massage and exits\n"
                        " -input:      Input file. Input file should under the input/ subdirectory\n"
                        "              Default is {} \n"
                        "              Hint: Only accepts OBJ files\n"
                        " -o:          JSON file output folder. Default is {} \n"
                        " -num_run:    Number of iterations for performance testing. Default is {} \n"                        
                        " -s:          Shuffle input. Default is false.\n"
                        " -p:          Sort input using patching output. Default is false.\n"
                        " -device_id:  GPU device ID. Default is {}",
            Arg.obj_file_name, Arg.output_folder, Arg.num_run, Arg.device_id);
            // clang-format on
            exit(EXIT_SUCCESS);
        }

        if (cmd_option_exists(argv, argc + argv, "-num_run")) {
            Arg.num_run = atoi(get_cmd_option(argv, argv + argc, "-num_run"));
        }

        if (cmd_option_exists(argv, argc + argv, "-input")) {
            Arg.obj_file_name =
                std::string(get_cmd_option(argv, argv + argc, "-input"));
        }
        if (cmd_option_exists(argv, argc + argv, "-o")) {
            Arg.output_folder =
                std::string(get_cmd_option(argv, argv + argc, "-o"));
        }
        if (cmd_option_exists(argv, argc + argv, "-device_id")) {
            Arg.device_id =
                atoi(get_cmd_option(argv, argv + argc, "-device_id"));
        }
        if (cmd_option_exists(argv, argc + argv, "-s")) {
            Arg.shuffle = true;
        }
        if (cmd_option_exists(argv, argc + argv, "-ev")) {
            Arg.ev = true;
        }
        if (cmd_option_exists(argv, argc + argv, "-vv")) {
            Arg.vv = true;
        }
    }

    RXMESH_TRACE("input= {}", Arg.obj_file_name);
    RXMESH_TRACE("output_folder= {}", Arg.output_folder);
    RXMESH_TRACE("num_run= {}", Arg.num_run);
    RXMESH_TRACE("device_id= {}", Arg.device_id);

    return RUN_ALL_TESTS();
}
