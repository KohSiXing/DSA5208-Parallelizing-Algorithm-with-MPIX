#include <algorithm>
#include <cctype>
#include <chrono>
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

template <class T> void free_vector(std::vector<T>& v) { std::vector<T>().swap(v); }

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ===========================================================================
// Data loading
// ===========================================================================

std::vector<Rating> read_ratings(const std::string& path, size_t max_rows = SIZE_MAX) {
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
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
        std::exit(1);
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

// Solve for every row of M:  (Y_r^T Y_r + lambda * n_r * I) x_r = Y_r^T m_r
void als_update(const CSR& M, const std::vector<double>& Y, std::vector<double>& X,
                int k, double lambda) {
    std::vector<double> A(k * k), b(k);
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
        std::copy(b.begin(), b.end(), X.begin() + (size_t)r * k);
    }
}

double predict(const std::vector<double>& U, const std::vector<double>& V, int u, int i, int k) {
    double s = 0;
    for (int f = 0; f < k; f++) s += U[(size_t)u * k + f] * V[(size_t)i * k + f];
    return s;
}

// Root mean squared error over the training ratings (stored in R, rows = users).
double train_rmse(const CSR& R, const std::vector<double>& U, const std::vector<double>& V, int k) {
    double se = 0;
    for (int r = 0; r < R.n_rows; r++)
        for (int p = R.ptr[r]; p < R.ptr[r + 1]; p++) {
            double e = R.val[p] - predict(U, V, r, R.idx[p], k);
            se += e * e;
        }
    return std::sqrt(se / R.idx.size());
}

double test_rmse(const std::vector<Rating>& test, const std::vector<double>& U,
                 const std::vector<double>& V, int k) {
    double se = 0;
    for (const auto& r : test) {
        double e = r.rating - predict(U, V, r.userId, r.movieId, k);
        se += e * e;
    }
    return std::sqrt(se / test.size());
}

void row_stats(const char* name, const CSR& M) {
    std::vector<int> cnt(M.n_rows);
    for (int r = 0; r < M.n_rows; r++) cnt[r] = M.ptr[r + 1] - M.ptr[r];
    std::sort(cnt.begin(), cnt.end());
    std::printf("ratings per %s: min %d, median %d, mean %.1f, max %d\n", name, cnt.front(),
                cnt[M.n_rows / 2], (double)M.idx.size() / M.n_rows, cnt.back());
}

// ===========================================================================
// Command-line options (same as the parallel programs)
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
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        std::fprintf(stderr,
            "usage: %s train_file [test_file] [-k 10] [-lambda 0.1] [-iters 10]\n"
            "                    [-rows N] [-sample p]\n"
            "       with one file, 20%% of its ratings are held out at random as the test set\n",
            argv[0]);
        return 1;
    }
    const int k = opt.k;

    // ---- 1. read and prepare the data ----
    auto t0 = Clock::now();
    std::vector<Rating> df = read_ratings(opt.train_file, opt.max_rows);
    sample_ratings(df, opt.sample, 7);
    std::vector<Rating> test;
    if (opt.test_file.empty()) {
        std::vector<Rating> train_part;
        split_train_test(df, train_part, test, 0.2, 123);
        df.swap(train_part);
    } else {
        test = read_ratings(opt.test_file);
    }
    auto uidx = make_index(df, true);
    auto iidx = make_index(df, false);
    int n_users = (int)uidx.size(), n_items = (int)iidx.size();
    std::vector<Rating> train = remap(df, uidx, iidx);
    free_vector(df);
    size_t n_test_raw = test.size();
    test = remap(test, uidx, iidx);
    CSR R  = build_csr(train, n_users, true);
    CSR Rt = build_csr(train, n_items, false);
    double mean = 0, se = 0;
    for (const auto& r : train) mean += r.rating;
    mean /= train.size();
    for (const auto& r : test) se += (r.rating - mean) * (r.rating - mean);
    size_t n_train = train.size();
    free_vector(train);
    double t_load = seconds_since(t0);

    std::printf("serial (1 core)\n");
    std::printf("train: %zu ratings, %d users, %d movies\n", n_train, n_users, n_items);
    std::printf("test : %zu ratings kept, %zu dropped (user/movie not in train)\n",
                test.size(), n_test_raw - test.size());
    row_stats("user ", R);
    row_stats("movie", Rt);
    std::printf("baseline (predict mean %.3f): test RMSE %.4f\n", mean, std::sqrt(se / test.size()));
    std::printf("\nsetup: load + prepare %.3f s\n", t_load);

    // ---- 2. initial factors (fixed seed, same as the parallel programs) ----
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> U((size_t)n_users * k), V((size_t)n_items * k);
    for (auto& x : U) x = dist(gen);
    for (auto& x : V) x = dist(gen);

    // ---- 3. ALS iterations ----
    std::printf("\nALS: k = %d, lambda = %g, serial\n", k, opt.lambda);
    std::printf(" iter   train RMSE    test RMSE   iter (s)\n");
    double sum_iter = 0, tr_rmse = 0, te_rmse = 0;
    int n_timed = 0;
    for (int it = 1; it <= opt.iters; it++) {
        auto ti = Clock::now();
        als_update(R, V, U, k, opt.lambda);      // fix V, solve for every user
        als_update(Rt, U, V, k, opt.lambda);     // fix U, solve for every movie
        double t_iter = seconds_since(ti);
        tr_rmse = train_rmse(R, U, V, k);        // not timed
        te_rmse = test_rmse(test, U, V, k);
        std::printf("%5d %12.6f %12.6f %10.4f\n", it, tr_rmse, te_rmse, t_iter);
        if (it > 1 || opt.iters == 1) { sum_iter += t_iter; n_timed++; }   // skip warm-up
    }

    // ---- 4. summary line (same columns as the parallel programs) ----
    double avg = sum_iter / n_timed;
    std::printf("\nRESULT,ranks,k,lambda,part,iters,load_s,distribute_s,avg_iter_s,avg_compute_s,"
                "avg_comm_s,avg_imbalance,train_rmse,test_rmse,threads,program,users,movies,"
                "train_ratings,avg_transfer_s\n");
    std::printf("RESULT,1,%d,%g,none,%d,%.3f,0,%.4f,%.4f,0,1,%.6f,%.6f,1,serial,%d,%d,%zu,0\n",
                k, opt.lambda, opt.iters, t_load, avg, avg, tr_rmse, te_rmse, n_users, n_items, n_train);
    return 0;
}
