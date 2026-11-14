#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Rating {
    int   userId;
    int   movieId;
    float rating;
    long  timestamp;
};

// Sparse matrix in CSR form. Row r's entries are positions [ptr[r], ptr[r+1]).
struct CSR {
    int n_rows = 0;
    std::vector<int>   ptr;   // n_rows + 1
    std::vector<int>   idx;   // column index of each entry (movie for R, user for Rt)
    std::vector<float> val;   // rating of each entry
};

// Release a vector's memory (clear() alone keeps the capacity).
template <class T> void free_vector(std::vector<T>& v) { std::vector<T>().swap(v); }

// ===========================================================================
// Data loading (same as the serial program; only rank 0 calls these)
// ===========================================================================

std::vector<Rating> read_ratings(const std::string& path, size_t max_rows = SIZE_MAX) {
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<Rating> df;
    std::string line;
    while (df.size() < max_rows && std::getline(file, line)) {
        if (line.empty() || !std::isdigit((unsigned char)line[0]))
            continue;                                   // skip header line
        for (char& c : line)                            // commas, tabs or "::" -> spaces
            if (c == ',' || c == '\t' || c == ':') c = ' ';
        std::stringstream ss(line);
        Rating r;
        if (ss >> r.userId >> r.movieId >> r.rating >> r.timestamp)
            df.push_back(r);
    }
    if (df.empty()) {
        std::fprintf(stderr, "no rows read from %s\n", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return df;
}

// Randomly put each rating in test with probability test_frac (fixed seed).
void split_train_test(const std::vector<Rating>& all, std::vector<Rating>& train,
                      std::vector<Rating>& test, double test_frac, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    train.clear();
    test.clear();
    for (const auto& r : all)
        (dist(gen) < test_frac ? test : train).push_back(r);
}

// Keep each rating with probability p (fixed seed). Used to change the number of
// ratings while keeping (almost) the same users and movies.
void sample_ratings(std::vector<Rating>& df, double p, unsigned seed) {
    if (p >= 1.0) return;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<Rating> kept;
    for (const auto& r : df)
        if (dist(gen) < p) kept.push_back(r);
    df.swap(kept);
}

// Original id -> compact 0-based index, assigned in order of first appearance.
std::unordered_map<int, int> make_index(const std::vector<Rating>& df, bool users) {
    std::unordered_map<int, int> index;
    for (const auto& r : df) {
        int id = users ? r.userId : r.movieId;
        if (!index.count(id)) {
            int next = (int)index.size();
            index[id] = next;
        }
    }
    return index;
}

// Replace ids with indices; drop rows whose user/movie isn't in the index.
std::vector<Rating> remap(const std::vector<Rating>& df,
                          const std::unordered_map<int, int>& uidx,
                          const std::unordered_map<int, int>& iidx) {
    std::vector<Rating> out;
    for (Rating r : df) {
        auto u = uidx.find(r.userId), i = iidx.find(r.movieId);
        if (u == uidx.end() || i == iidx.end()) continue;
        r.userId = u->second;
        r.movieId = i->second;
        out.push_back(r);
    }
    return out;
}

// Build CSR grouped by user (by_user = true) or by movie (false).
CSR build_csr(const std::vector<Rating>& df, int n_rows, bool by_user) {
    CSR M;
    M.n_rows = n_rows;
    M.ptr.assign(n_rows + 1, 0);
    for (const auto& r : df) M.ptr[(by_user ? r.userId : r.movieId) + 1]++;
    for (int i = 0; i < n_rows; i++) M.ptr[i + 1] += M.ptr[i];
    M.idx.resize(df.size());
    M.val.resize(df.size());
    std::vector<int> next(M.ptr.begin(), M.ptr.end() - 1);
    for (const auto& r : df) {
        int row = by_user ? r.userId : r.movieId;
        int p = next[row]++;
        M.idx[p] = by_user ? r.movieId : r.userId;
        M.val[p] = r.rating;
    }
    return M;
}

// ===========================================================================
// ALS Formula
// Factors are stored row-major: X[r*k + f] is factor f of row r.
// ===========================================================================

// Solve A x = b for a symmetric positive-definite k x k matrix A (Cholesky).
// A is overwritten with its factor L; b is overwritten with the solution x.
void cholesky_solve(std::vector<double>& A, std::vector<double>& b, int k) {
    for (int j = 0; j < k; j++) {                     // A = L L^T
        double d = A[j * k + j];
        for (int p = 0; p < j; p++) d -= A[j * k + p] * A[j * k + p];
        d = std::sqrt(d);
        A[j * k + j] = d;
        for (int i = j + 1; i < k; i++) {
            double s = A[i * k + j];
            for (int p = 0; p < j; p++) s -= A[i * k + p] * A[j * k + p];
            A[i * k + j] = s / d;
        }
    }
    for (int i = 0; i < k; i++) {                     // L y = b
        double s = b[i];
        for (int p = 0; p < i; p++) s -= A[i * k + p] * b[p];
        b[i] = s / A[i * k + i];
    }
    for (int i = k - 1; i >= 0; i--) {                // L^T x = y
        double s = b[i];
        for (int p = i + 1; p < k; p++) s -= A[p * k + i] * b[p];
        b[i] = s / A[i * k + i];
    }
}

// Solve for every row of this rank's block M:
//   (Y_r^T Y_r + lambda * n_r * I) x_r = Y_r^T m_r
// Local row r is global row row0 + r, so its answer goes to row row0 + r of X.
// The rows are shared among the OpenMP threads; each thread has its own A and b.
void als_update(const CSR& M, int row0, const std::vector<double>& Y, std::vector<double>& X,
                int k, double lambda) {
    #pragma omp parallel
    {
    std::vector<double> A(k * k), b(k);          // private to this thread
    #pragma omp for schedule(dynamic, 16)
    for (int r = 0; r < M.n_rows; r++) {
        std::fill(A.begin(), A.end(), 0.0);
        std::fill(b.begin(), b.end(), 0.0);
        int n_r = M.ptr[r + 1] - M.ptr[r];
        for (int p = M.ptr[r]; p < M.ptr[r + 1]; p++) {
            const double* y = &Y[(size_t)M.idx[p] * k];
            for (int a = 0; a < k; a++) {
                b[a] += M.val[p] * y[a];
                for (int c = 0; c <= a; c++) A[a * k + c] += y[a] * y[c];   // lower triangle
            }
        }
        for (int a = 0; a < k; a++) {
            for (int c = 0; c < a; c++) A[c * k + a] = A[a * k + c];       // mirror to upper
            A[a * k + a] += lambda * std::max(n_r, 1);
        }
        cholesky_solve(A, b, k);
        std::copy(b.begin(), b.end(), X.begin() + (size_t)(row0 + r) * k);
    }
    }   // end of omp parallel
}

double predict(const std::vector<double>& U, const std::vector<double>& V, int u, int i, int k) {
    double s = 0;
    for (int f = 0; f < k; f++) s += U[(size_t)u * k + f] * V[(size_t)i * k + f];
    return s;
}

// Sum of squared errors over this rank's training ratings (its block of R; rows are users).
double train_sq_error(const CSR& R, int u0, const std::vector<double>& U,
                      const std::vector<double>& V, int k) {
    double se = 0;
    #pragma omp parallel for reduction(+ : se) schedule(dynamic, 64)
    for (int r = 0; r < R.n_rows; r++)
        for (int p = R.ptr[r]; p < R.ptr[r + 1]; p++) {
            double e = R.val[p] - predict(U, V, u0 + r, R.idx[p], k);
            se += e * e;
        }
    return se;
}

// Sum of squared errors over this rank's test ratings.
double test_sq_error(const std::vector<int>& tu, const std::vector<int>& ti,
                     const std::vector<float>& tr, const std::vector<double>& U,
                     const std::vector<double>& V, int k) {
    double se = 0;
    #pragma omp parallel for reduction(+ : se)
    for (long long j = 0; j < (long long)tu.size(); j++) {
        double e = tr[j] - predict(U, V, tu[j], ti[j], k);
        se += e * e;
    }
    return se;
}

// ===========================================================================
// Splitting the rows across ranks, and sending each rank its part
// ===========================================================================

// Split rows [0, n_rows) into P contiguous blocks. Returns P+1 boundaries:
// rank p owns rows [bounds[p], bounds[p+1]).
//   by_nnz = true : each block holds about the same number of ratings
//   by_nnz = false: each block holds the same number of rows
std::vector<int> make_partition(const CSR& M, int P, bool by_nnz) {
    std::vector<int> bounds(P + 1);
    long long total = M.ptr[M.n_rows];
    for (int p = 0; p <= P; p++) {
        if (!by_nnz) {
            bounds[p] = (int)((long long)M.n_rows * p / P);
        } else {
            long long target = total * p / P;      // first row whose start reaches target
            bounds[p] = (int)(std::lower_bound(M.ptr.begin(), M.ptr.end(), target) - M.ptr.begin());
        }
    }
    bounds[0] = 0;
    bounds[P] = M.n_rows;
    return bounds;
}

// Rank 0 holds the full matrix G; every rank receives its own rows
// [bounds[rank], bounds[rank+1]) as a local CSR (local row 0 = global row bounds[rank]).
CSR scatter_rows(const CSR& G, const std::vector<int>& bounds, int rank, int P) {
    // rows per rank and where they start
    std::vector<int> row_counts(P), row_displs(P);
    for (int p = 0; p < P; p++) {
        row_displs[p] = bounds[p];
        row_counts[p] = bounds[p + 1] - bounds[p];
    }
    // ratings per rank and where they start (only rank 0 can compute these)
    std::vector<int> nnz_counts(P, 0), nnz_displs(P, 0), row_lengths;
    if (rank == 0) {
        for (int p = 0; p < P; p++) {
            nnz_displs[p] = G.ptr[bounds[p]];
            nnz_counts[p] = G.ptr[bounds[p + 1]] - G.ptr[bounds[p]];
        }
        row_lengths.resize(G.n_rows);
        for (int r = 0; r < G.n_rows; r++) row_lengths[r] = G.ptr[r + 1] - G.ptr[r];
    }
    int my_nnz = 0;
    MPI_Scatter(nnz_counts.data(), 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CSR L;
    L.n_rows = row_counts[rank];
    std::vector<int> my_lengths(L.n_rows);
    MPI_Scatterv(row_lengths.data(), row_counts.data(), row_displs.data(), MPI_INT,
                 my_lengths.data(), L.n_rows, MPI_INT, 0, MPI_COMM_WORLD);
    L.ptr.assign(L.n_rows + 1, 0);
    for (int r = 0; r < L.n_rows; r++) L.ptr[r + 1] = L.ptr[r] + my_lengths[r];

    L.idx.resize(my_nnz);
    L.val.resize(my_nnz);
    MPI_Scatterv(G.idx.data(), nnz_counts.data(), nnz_displs.data(), MPI_INT,
                 L.idx.data(), my_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(G.val.data(), nnz_counts.data(), nnz_displs.data(), MPI_FLOAT,
                 L.val.data(), my_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return L;
}

// Rank 0 holds all test ratings; every rank receives the test ratings of the
// users it owns, as three arrays (user, movie, rating).
void scatter_test(const std::vector<Rating>& test, const std::vector<int>& ubounds, int rank,
                  int P, std::vector<int>& tu, std::vector<int>& ti, std::vector<float>& tr) {
    std::vector<int> counts(P, 0), displs(P, 0);
    std::vector<int> all_u, all_i;
    std::vector<float> all_r;
    if (rank == 0) {
        // owner of user u = the block p with bounds[p] <= u < bounds[p+1]
        std::vector<int> owner(test.size());
        for (size_t j = 0; j < test.size(); j++) {
            owner[j] = (int)(std::upper_bound(ubounds.begin(), ubounds.end(), test[j].userId)
                             - ubounds.begin()) - 1;
            counts[owner[j]]++;
        }
        for (int p = 1; p < P; p++) displs[p] = displs[p - 1] + counts[p - 1];
        all_u.resize(test.size());
        all_i.resize(test.size());
        all_r.resize(test.size());
        std::vector<int> next = displs;
        for (size_t j = 0; j < test.size(); j++) {
            int q = next[owner[j]]++;
            all_u[q] = test[j].userId;
            all_i[q] = test[j].movieId;
            all_r[q] = test[j].rating;
        }
    }
    int my_n = 0;
    MPI_Scatter(counts.data(), 1, MPI_INT, &my_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tu.resize(my_n);
    ti.resize(my_n);
    tr.resize(my_n);
    MPI_Scatterv(all_u.data(), counts.data(), displs.data(), MPI_INT, tu.data(), my_n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_i.data(), counts.data(), displs.data(), MPI_INT, ti.data(), my_n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_r.data(), counts.data(), displs.data(), MPI_FLOAT, tr.data(), my_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// Print the min / median / mean / max number of ratings per row.
void row_stats(const char* name, const CSR& M) {
    std::vector<int> cnt(M.n_rows);
    for (int r = 0; r < M.n_rows; r++) cnt[r] = M.ptr[r + 1] - M.ptr[r];
    std::sort(cnt.begin(), cnt.end());
    std::printf("ratings per %s: min %d, median %d, mean %.1f, max %d\n", name, cnt.front(),
                cnt[M.n_rows / 2], (double)M.idx.size() / M.n_rows, cnt.back());
}

// ===========================================================================
// Command-line options
// ===========================================================================

struct Options {
    std::string train_file, test_file;    // test_file empty -> random 80/20 split
    size_t max_rows = SIZE_MAX;
    int    k = 10;
    double lambda = 0.1;
    int    iters = 10;
    bool   by_nnz = true;
    double sample = 1.0;
};

bool parse_args(int argc, char** argv, Options& o) {
    std::vector<std::string> files;
    for (int a = 1; a < argc; a++) {
        std::string s = argv[a];
        bool has_value = (a + 1 < argc);
        if      (s == "-k"      && has_value) o.k = std::atoi(argv[++a]);
        else if (s == "-lambda" && has_value) o.lambda = std::atof(argv[++a]);
        else if (s == "-iters"  && has_value) o.iters = std::atoi(argv[++a]);
        else if (s == "-rows"   && has_value) o.max_rows = std::stoull(argv[++a]);
        else if (s == "-sample" && has_value) o.sample = std::atof(argv[++a]);
        else if (s == "-part"   && has_value) {
            std::string v = argv[++a];
            if (v != "nnz" && v != "rows") return false;
            o.by_nnz = (v == "nnz");
        }
        else if (!s.empty() && s[0] == '-') return false;
        else files.push_back(s);
    }
    if (files.empty() || files.size() > 2 || o.k < 1 || o.iters < 1) return false;
    if (o.sample <= 0 || o.sample > 1) return false;
    o.train_file = files[0];
    if (files.size() == 2) o.test_file = files[1];
    return true;
}

// ===========================================================================

int main(int argc, char** argv) {
    // Only the main thread of each rank makes MPI calls.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    const int T = omp_get_max_threads();          // OpenMP threads per rank
    if (provided < MPI_THREAD_FUNNELED && rank == 0)
        std::fprintf(stderr, "warning: MPI library does not support MPI_THREAD_FUNNELED\n");

    Options opt;
    if (!parse_args(argc, argv, opt)) {
        if (rank == 0)
            std::fprintf(stderr,
                "usage: mpirun -np P %s train_file [test_file] [-k 10] [-lambda 0.1] [-iters 10]\n"
                "                    [-part nnz|rows] [-rows N] [-sample p]\n"
                "       with one file, 20%% of its ratings are held out at random as the test set\n",
                argv[0]);
        MPI_Finalize();
        return 1;
    }
    const int k = opt.k;

    // ---- 1. rank 0 reads and prepares the data (same steps as the serial code) ----
    double t_setup0 = MPI_Wtime();
    int n_users = 0, n_items = 0;
    CSR R, Rt;                        // full matrices, rank 0 only
    std::vector<Rating> test;         // full test set, rank 0 only
    if (rank == 0) {
        std::vector<Rating> df = read_ratings(opt.train_file, opt.max_rows);
        sample_ratings(df, opt.sample, 7);
        if (opt.test_file.empty()) {
            std::vector<Rating> train_part;
            split_train_test(df, train_part, test, 0.2, 123);
            df.swap(train_part);
        } else {
            test = read_ratings(opt.test_file);
        }
        auto uidx = make_index(df, true);
        auto iidx = make_index(df, false);
        n_users = (int)uidx.size();
        n_items = (int)iidx.size();
        std::vector<Rating> train = remap(df, uidx, iidx);
        free_vector(df);
        size_t n_test_raw = test.size();
        test = remap(test, uidx, iidx);
        R  = build_csr(train, n_users, true);
        Rt = build_csr(train, n_items, false);

        double mean = 0, se = 0;
        for (const auto& r : train) mean += r.rating;
        mean /= train.size();
        for (const auto& r : test) se += (r.rating - mean) * (r.rating - mean);
        free_vector(train);

        std::printf("MPI ranks: %d, OpenMP threads per rank: %d (%d cores in total)\n", P, T, P * T);
        std::printf("train: %zu ratings, %d users, %d movies\n", R.idx.size(), n_users, n_items);
        std::printf("test : %zu ratings kept, %zu dropped (user/movie not in train)\n",
                    test.size(), n_test_raw - test.size());
        row_stats("user ", R);
        row_stats("movie", Rt);
        std::printf("baseline (predict mean %.3f): test RMSE %.4f\n", mean, std::sqrt(se / test.size()));
    }
    double t_load = MPI_Wtime() - t_setup0;

    // ---- 2. decide which users and movies each rank owns ----
    MPI_Bcast(&n_users, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_items, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> ub(P + 1), ib(P + 1);         // user / movie block boundaries
    if (rank == 0) {
        ub = make_partition(R, P, opt.by_nnz);
        ib = make_partition(Rt, P, opt.by_nnz);

        std::printf("\npartition (-part %s):\n", opt.by_nnz ? "nnz" : "rows");
        std::printf(" rank      users   user ratings     movies  movie ratings\n");
        long long max_u = 0, max_i = 0;
        for (int p = 0; p < P; p++) {
            long long nu = R.ptr[ub[p + 1]] - R.ptr[ub[p]];
            long long ni = Rt.ptr[ib[p + 1]] - Rt.ptr[ib[p]];
            max_u = std::max(max_u, nu);
            max_i = std::max(max_i, ni);
            std::printf("%5d %10d %14lld %10d %14lld\n", p, ub[p + 1] - ub[p], nu, ib[p + 1] - ib[p], ni);
        }
        double avg = (double)R.idx.size() / P;
        std::printf("largest block / average: users %.3f, movies %.3f\n", max_u / avg, max_i / avg);
    }
    MPI_Bcast(ub.data(), P + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(ib.data(), P + 1, MPI_INT, 0, MPI_COMM_WORLD);
    const int u0 = ub[rank], i0 = ib[rank];

    // ---- 3. send every rank its rows of R and Rt, and its users' test ratings ----
    double t_dist0 = MPI_Wtime();
    CSR R_loc  = scatter_rows(R, ub, rank, P);
    CSR Rt_loc = scatter_rows(Rt, ib, rank, P);
    std::vector<int> tu, ti;
    std::vector<float> tr;
    scatter_test(test, ub, rank, P, tu, ti, tr);
    if (rank == 0) {                              // rank 0 no longer needs the full copies
        free_vector(R.ptr);  free_vector(R.idx);  free_vector(R.val);
        free_vector(Rt.ptr); free_vector(Rt.idx); free_vector(Rt.val);
        free_vector(test);
    }
    double t_dist = MPI_Wtime() - t_dist0;

    // global counts for the RMSE
    double counts_loc[2] = {(double)R_loc.idx.size(), (double)tu.size()}, counts[2];
    MPI_Allreduce(counts_loc, counts, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // ---- 4. initial factors: every rank generates the same full U and V ----
    // (same seed and order as the serial code, so the results can be compared)
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> U((size_t)n_users * k), V((size_t)n_items * k);
    for (auto& x : U) x = dist(gen);
    for (auto& x : V) x = dist(gen);

    // how many numbers each rank contributes to U and V, and where they go
    std::vector<int> u_counts(P), u_displs(P), i_counts(P), i_displs(P);
    for (int p = 0; p < P; p++) {
        u_counts[p] = (ub[p + 1] - ub[p]) * k;  u_displs[p] = ub[p] * k;
        i_counts[p] = (ib[p + 1] - ib[p]) * k;  i_displs[p] = ib[p] * k;
    }

    if (rank == 0) {
        std::printf("\nsetup: load + prepare %.3f s (rank 0), distribute %.3f s\n", t_load, t_dist);
        std::printf("\nALS: k = %d, lambda = %g, %d ranks x %d threads\n", k, opt.lambda, P, T);
        std::printf(" iter   train RMSE    test RMSE   iter (s)  compute (s)  comm (s)  imbalance\n");
    }

    // ---- 5. ALS iterations ----
    double sum_iter = 0, sum_comp = 0, sum_comm = 0, sum_imb = 0;
    int n_timed = 0;
    double train_rmse = 0, test_rmse = 0;
    for (int it = 1; it <= opt.iters; it++) {
        double t0 = MPI_Wtime();
        als_update(R_loc, u0, V, U, k, opt.lambda);              // my users
        double t1 = MPI_Wtime();
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,       // share U
                       U.data(), u_counts.data(), u_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        als_update(Rt_loc, i0, U, V, k, opt.lambda);             // my movies
        double t3 = MPI_Wtime();
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,       // share V
                       V.data(), i_counts.data(), i_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        double t4 = MPI_Wtime();

        // "comm" = time inside MPI calls, which includes waiting for slower ranks
        double mine[3] = {t4 - t0, (t1 - t0) + (t3 - t2), (t2 - t1) + (t4 - t3)};
        double worst[3], comp_sum = 0;
        MPI_Reduce(mine, worst, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mine[1], &comp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // RMSE (not timed): each rank sums its own squared errors, then add them up
        double se_loc[2] = {train_sq_error(R_loc, u0, U, V, k), test_sq_error(tu, ti, tr, U, V, k)};
        double se[2];
        MPI_Reduce(se_loc, se, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            train_rmse = std::sqrt(se[0] / counts[0]);
            test_rmse  = std::sqrt(se[1] / counts[1]);
            double imbalance = worst[1] / (comp_sum / P);        // slowest rank vs average
            std::printf("%5d %12.6f %12.6f %10.4f %12.4f %9.4f %10.3f\n", it, train_rmse, test_rmse,
                        worst[0], worst[1], worst[2], imbalance);
            if (it > 1 || opt.iters == 1) {                      // skip warm-up iteration
                sum_iter += worst[0]; sum_comp += worst[1]; sum_comm += worst[2];
                sum_imb += imbalance; n_timed++;
            }
        }
    }

    // ---- 6. one summary line, easy to paste into a spreadsheet ----
    if (rank == 0) {
        std::printf("\nRESULT,ranks,k,lambda,part,iters,load_s,distribute_s,avg_iter_s,avg_compute_s,"
                    "avg_comm_s,avg_imbalance,train_rmse,test_rmse,threads,program,users,movies,"
                    "train_ratings,avg_transfer_s\n");
        std::printf("RESULT,%d,%d,%g,%s,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%.6f,%.6f,%d,hybrid,%d,%d,%.0f,0\n",
                    P, k, opt.lambda, opt.by_nnz ? "nnz" : "rows", opt.iters, t_load, t_dist,
                    sum_iter / n_timed, sum_comp / n_timed, sum_comm / n_timed, sum_imb / n_timed,
                    train_rmse, test_rmse, T, n_users, n_items, counts[0]);
    }

    MPI_Finalize();
    return 0;
}
