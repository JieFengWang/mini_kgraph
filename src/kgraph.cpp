
#include "kgraph.h"

KGraphConstructor::KGraphConstructor(IndexOracle const &o, IndexParams &p)
    : oracle(o), params(p), nhoods(o.size()), n_comps(0)
{
}

KGraphConstructor::~KGraphConstructor()
{
}

void KGraphConstructor::init()
{

    unsigned N = oracle.size();
    unsigned seed = params.seed;
    mt19937 rng(seed);
    for (auto &nhood : nhoods)
    {
        nhood.nn_new.resize(params.S * 2);
        nhood.pool.resize(params.L + 1);
        nhood.radius = numeric_limits<float>::max();
    }
#pragma omp parallel
    {
#ifdef _OPENMP
        mt19937 rng(seed ^ omp_get_thread_num());
#else
        mt19937 rng(seed);
#endif
        vector<unsigned> random(params.S + 1);
#pragma omp for
        for (unsigned n = 0; n < N; ++n)
        {
            auto &nhood = nhoods[n];
            Neighbors &pool = nhood.pool;
            GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
            GenRandom(rng, &random[0], random.size(), N);
            nhood.L = params.S;
            nhood.M = params.S;
            unsigned i = 0;
            for (unsigned l = 0; l < nhood.L; ++l)
            {
                if (random[i] == n)
                    ++i;
                auto &nn = nhood.pool[l];
                nn.id = random[i++];
                nn.dist = oracle(nn.id, n);
                nn.flag = true;
            }
            sort(pool.begin(), pool.begin() + nhood.L);
        }
    }
}

void KGraphConstructor::join()
{
    size_t cc = 0;
#pragma omp parallel for default(shared) schedule(dynamic, 100) reduction(+ : cc)
    for (unsigned n = 0; n < oracle.size(); ++n)
    {
        size_t uu = 0;
        nhoods[n].found = false;
        nhoods[n].join([&](unsigned i, unsigned j)
                       {
                float dist = oracle(i, j);
                ++cc;
                unsigned r;
                r = nhoods[i].parallel_try_insert(j, dist);
                if (r < params.K) ++uu;
                nhoods[j].parallel_try_insert(i, dist);
                if (r < params.K) ++uu; });
        nhoods[n].found = uu > 0;
    }
    n_comps += cc;
}

void KGraphConstructor::update()
{
    unsigned N = oracle.size();
    for (auto &nhood : nhoods)
    {
        nhood.nn_new.clear();
        nhood.nn_old.clear();
        nhood.rnn_new.clear();
        nhood.rnn_old.clear();
        nhood.radius = nhood.pool.back().dist;
    }
    //!!! compute radius2
#pragma omp parallel for
    for (unsigned n = 0; n < N; ++n)
    {
        auto &nhood = nhoods[n];
        if (nhood.found)
        {
            unsigned maxl = std::min(nhood.M + params.S, nhood.L);
            unsigned c = 0;
            unsigned l = 0;
            while ((l < maxl) && (c < params.S))
            {
                if (nhood.pool[l].flag)
                    ++c;
                ++l;
            }
            nhood.M = l;
        }
        BOOST_VERIFY(nhood.M > 0);
        nhood.radiusM = nhood.pool[nhood.M - 1].dist;
    }
#pragma omp parallel for
    for (unsigned n = 0; n < N; ++n)
    {
        auto &nhood = nhoods[n];
        auto &nn_new = nhood.nn_new;
        auto &nn_old = nhood.nn_old;
        for (unsigned l = 0; l < nhood.M; ++l)
        {
            auto &nn = nhood.pool[l];
            auto &nhood_o = nhoods[nn.id];
            if (nn.flag)
            {
                nn_new.push_back(nn.id);
                if (nn.dist > nhood_o.radiusM)
                {
                    LockGuard guard(nhood_o.lock);
                    nhood_o.rnn_new.push_back(n);
                }
                nn.flag = false;
            }
            else
            {
                nn_old.push_back(nn.id);
                if (nn.dist > nhood_o.radiusM)
                {
                    LockGuard guard(nhood_o.lock);
                    nhood_o.rnn_old.push_back(n);
                }
            }
        }
    }
    for (unsigned i = 0; i < N; ++i)
    {
        auto &nn_new = nhoods[i].nn_new;
        auto &nn_old = nhoods[i].nn_old;
        auto &rnn_new = nhoods[i].rnn_new;
        auto &rnn_old = nhoods[i].rnn_old;
        if (params.R && (rnn_new.size() > params.R))
        {
            random_shuffle(rnn_new.begin(), rnn_new.end());
            rnn_new.resize(params.R);
        }
        nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
        if (params.R && (rnn_old.size() > params.R))
        {
            random_shuffle(rnn_old.begin(), rnn_old.end());
            rnn_old.resize(params.R);
        }
        nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
    }
}

void KGraphConstructor::build_index()
{

    {
        if (params.S > params.L)
        {
            params.S = params.L;
            params.R = params.L;
        }
    }

    unsigned N = oracle.size();
    cout << "N = " << N << "\n";

    init();

    float total = N * float(N - 1) / 2;
    for (unsigned it = 0; (it < params.iterations); ++it)
    {
        cout << "iter ... " << (it + 1) << endl;
        join();
        update();
        cout << "after iter " << (it + 1) << "\n";
    }
}

void KGraphConstructor::inner_save(string path, unsigned K)
{
    vector<Nhood> &nbh = nhoods;

    if (K == 0)
        K = 10;

    unsigned int size_n = nbh.size();
    unsigned int dim = K;

    auto myfile = std::fstream(path, ios::out | ios::binary);
    for (int i = 0; i < size_n; ++i)
    {
        auto const &pool = nbh[i].pool;
        myfile.write((char *)&dim, sizeof(unsigned int));
        for (int j = 0; j < dim; ++j)
        {
            myfile.write((char *)&pool[j].id, sizeof(unsigned int));
        }
    }
    myfile.close();
    cout << path << " has been saved!" << endl;
}

template <typename RNG>
void KGraphConstructor::GenRandom(RNG &rng, unsigned *addr, unsigned size, unsigned N)
{
    if (N == size)
    {
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = i;
        }
        return;
    }
    for (unsigned i = 0; i < size; ++i)
    {
        addr[i] = rng() % (N - size);
    }
    sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i)
    {
        if (addr[i] <= addr[i - 1])
        {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i)
    {
        addr[i] = (addr[i] + off) % N;
    }
}

float KGraphConstructor::evaluate(vector<vector<unsigned int>> &gt)
{
    auto size_n = gt.size();
    auto check_k = 10; // min(para_.K, recallN);
    unsigned int hit = 0.0;

    for (unsigned int i = 0; i < size_n; ++i)
    {
        auto &pool = nhoods[i].pool;
        for (unsigned int j = 0; j < check_k; ++j)
        {
            auto res_idx = pool[j].id;
            for (unsigned int k = 0; k < check_k; ++k)
            {
                auto gt_idx = gt[i][k];
                if (fabs((float)res_idx - (float)gt_idx) < 1E-4)
                {
                    hit += 1; // atmic op
                    break;
                }
            }
        }
    }
    return (float)1.0 * hit / (size_n * check_k);
}
