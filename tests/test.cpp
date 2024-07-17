#include <iostream>
#include <omp.h>
#include <chrono>

#include "../src/io.hpp"
#include "../src/kgraph.h"

using namespace std;

int main(int argc, char *argv[])
{

    std::string base_path = "./data/sift1m_base.fvecs";

    IndexParams params;

    params.L = 100;         //// K of k-NN Graph  unsigned K = params.L;
    params.R = 16;          // 32; /// smpN
    params.S = 16;          // 32;//// smpN
    params.K = 10;          //// search K && recall K
    params.iterations = 10; // iterN
    params.controls = 0;
    params.reverse = 0;

    std::string save_pth;
    unsigned save_topk = 10;
    if (argc > 1)
    {
        base_path = argv[1];
    }
    if (argc > 2)
    {
        save_pth = argv[2];
    }

    Matrix<float> base_data;
    base_data.load(base_path, 128, 0, 4);

    MatrixOracle<float, metric::l2sqr> oracle(base_data);

    KGraphConstructor *kg = new KGraphConstructor(oracle, params);

    auto start = chrono::high_resolution_clock::now();
    kg->build_index();
    auto end = chrono::high_resolution_clock ::now();

    cout << "Elapsed time in milliseconds: "
         << 1.0 * chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000
         << " s" << endl;

    if (save_pth.size() > 2)
        kg->inner_save(save_pth, save_topk);

    delete kg;

    return 0;
}
