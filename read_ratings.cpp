// read_ratings.cpp - read MovieLens ratings and print them like a dataframe.
// Handles ml-100k u.data / u1.base (tabs), ml-1m ratings.dat (::)
// and ml-32m ratings.csv (commas, with header).
//
// Build: g++ -O2 -std=c++17 -o read_ratings read_ratings.cpp
// Run:   ./read_ratings ml-100k/u.data
//        ./read_ratings ml-32m/ratings.csv 100000   (read first 100k rows only)

#include <algorithm>
#include <vector>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <random>

struct Rating {
    int    userId;
    int    movieId;
    float  rating;
    long   timestamp;
};

struct CSR {
    int n_rows = 0;
    std::vector<int>   ptr;   // n_rows + 1
    std::vector<int>   idx;   // column index of each entry (movie for R, user for Rt)
    std::vector<float> val;   // rating of each entry
};

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Print rows [begin, end) as a table
void print_rows(const std::vector<Rating>& df, size_t begin, size_t end) {
    std::cout << std::setw(10) << "" << std::setw(10) << "userId" << std::setw(10)
              << "movieId" << std::setw(10) << "rating" << std::setw(14) << "timestamp" << "\n";
    for (size_t i = begin; i < end; ++i) {
        std::cout << std::setw(10) << i << std::setw(10) << df[i].userId << std::setw(10)
                  << df[i].movieId << std::setw(10) << std::fixed << std::setprecision(1)
                  << df[i].rating << std::setw(14) << df[i].timestamp << "\n";
    }
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

// Min / median / mean / max ratings per row.
void row_stats(const std::string& name, const CSR& M) {
    std::vector<int> cnt(M.n_rows);
    for (int r = 0; r < M.n_rows; r++) cnt[r] = M.ptr[r + 1] - M.ptr[r];
    std::sort(cnt.begin(), cnt.end());
    std::cout << "ratings per " << name << ": min " << cnt.front()
              << ", median " << cnt[M.n_rows / 2]
              << ", mean " << std::setprecision(1) << (double)M.idx.size() / M.n_rows
              << ", max " << cnt.back() << "\n";
}

// ---------------------------------------------------------------------------
// ALS
// Factors are stored row-major: X[r*k + f] is factor f of row r.
// ---------------------------------------------------------------------------

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

// One half-step of ALS: for every row r of M, solve
//   (Y_r^T Y_r + lambda * n_r * I) x_r = Y_r^T m_r
// Called as als_update(R, V, U) for users and als_update(Rt, U, V) for movies.
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

// Root mean squared error of the predictions U[u] . V[i] over a set of ratings.
double rmse(const std::vector<Rating>& data, const std::vector<double>& U,
            const std::vector<double>& V, int k) {
    double se = 0;
    for (const auto& r : data) {
        double pred = 0;
        for (int f = 0; f < k; f++) pred += U[(size_t)r.userId * k + f] * V[(size_t)r.movieId * k + f];
        double err = r.rating - pred;
        se += err * err;
    }
    return std::sqrt(se / data.size());
}

std::vector<Rating> read_ratings(const std::string& path, size_t max_rows = SIZE_MAX) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }

    std::vector<Rating> df;
    std::string line;

    while (df.size() < max_rows && std::getline(file, line)) {
        if (line.empty() || !std::isdigit((unsigned char)line[0]))
            continue;                         // skip header line (ratings.csv has one, u.data doesn't)

        // Works for various delimiters - comma, tabs, colon, space
        for (char& c : line)
            if (c == ',' || c == '\t' || c == ':') c = ' ';

        std::stringstream ss(line);
        Rating r;
        if (ss >> r.userId >> r.movieId >> r.rating >> r.timestamp)
            df.push_back(r);
        else
            std::cerr << "skipping bad line: " << line << "\n";
    }

    if (df.empty()) {
        std::cerr << "no rows read\n";
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
    train.reserve((size_t)(all.size() * (1 - test_frac) * 1.01));
    test.reserve((size_t)(all.size() * test_frac * 1.01));
    for (const auto& r : all)
        (dist(gen) < test_frac ? test : train).push_back(r);
}

// Release a vector's memory (clear() alone keeps the capacity).
template <class T> void free_vector(std::vector<T>& v) { std::vector<T>().swap(v); }

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " train_file test_file\n"
                  << "       " << argv[0] << " ratings_file [max_rows]   (random 80/20 split)\n";
        return 1;
    }
    // Second argument is either a test file or a row limit.
    bool split_mode = (argc < 3) || std::isdigit((unsigned char)argv[2][0]);
    size_t max_rows = (split_mode && argc >= 3) ? std::stoul(argv[2]) : SIZE_MAX;

    auto t0 = Clock::now();
    std::vector<Rating> df = read_ratings(argv[1], max_rows);
    std::vector<Rating> test;
    if (!split_mode) test = read_ratings(argv[2]);
    double t_read = seconds_since(t0);


    // df.head() / df.tail()
    size_t n = df.size();
    std::cout << "head:\n";
    print_rows(df, 0, std::min<size_t>(5, n));
    std::cout << "...\ntail:\n";
    print_rows(df, n - std::min<size_t>(5, n), n);

    // df.shape
    std::cout << "\nshape: (" << n << ", 4)\n";

    // a little of df.describe()
    std::unordered_set<int> users, movies;
    double sum = 0;
    float rmin = df[0].rating, rmax = df[0].rating;
    for (const auto& r : df) {
        users.insert(r.userId);
        movies.insert(r.movieId);
        sum += r.rating;
        rmin = std::min(rmin, r.rating);
        rmax = std::max(rmax, r.rating);
    }
    std::cout << "unique users : " << users.size() << "\n"
              << "unique movies: " << movies.size() << "\n"
              << "rating min / mean / max: " << std::setprecision(1) << rmin << " / "
              << std::setprecision(3) << sum / n << " / " << std::setprecision(1) << rmax << "\n";

    // rating histogram
    std::cout << "rating histogram:\n";
    for (float v = rmin; v <= rmax + 1e-6f; v += 0.5f) {
        size_t c = std::count_if(df.begin(), df.end(),
                                 [v](const Rating& r) { return std::abs(r.rating - v) < 1e-6f; });
        if (c) std::cout << "  " << std::setprecision(1) << v << " : " << c << "\n";
    }

    // one file given: hold out 20% of ratings as the test set
    if (split_mode) {
        std::vector<Rating> train_part;
        split_train_test(df, train_part, test, 0.2, 123);
        df.swap(train_part);          // df is now the training part
        free_vector(train_part);      // free the full copy
        std::cout << "\nrandom split: " << df.size() << " train, " << test.size() << " test\n";
    }

    // ids -> 0-based indices (from train only), then build R and Rt
    t0 = Clock::now();
    auto uidx = make_index(df, true);
    auto iidx = make_index(df, false);
    int n_users = (int)uidx.size(), n_items = (int)iidx.size();
    std::vector<Rating> train = remap(df, uidx, iidx);
    free_vector(df); 
    size_t n_test_raw = test.size();
    test = remap(test, uidx, iidx);
    CSR R  = build_csr(train, n_users, true);
    CSR Rt = build_csr(train, n_items, false);
    double t_build = seconds_since(t0);

    std::cout << "\ntrain: " << train.size() << " ratings, " << n_users << " users, "
              << n_items << " movies\n";
    std::cout << "test : " << test.size() << " ratings kept, "
              << n_test_raw - test.size() << " dropped (user/movie not in train)\n";
    row_stats("user ", R);
    row_stats("movie", Rt);

    std::cout << std::setprecision(4) << "\ntime: read " << t_read << " s, index + CSR "
              << t_build << " s\n";

    // ALS 
    int    k      = 10;      // number of latent factors
    double lambda = 0.1;     // regularisation strength
    int    iters  = 10;

    // baseline: always predict the training mean
    double mean = 0;
    for (const auto& r : train) mean += r.rating;
    mean /= train.size();
    double se = 0;
    for (const auto& r : test) se += (r.rating - mean) * (r.rating - mean);
    std::cout << "\nbaseline (predict mean " << std::setprecision(3) << mean
              << "): test RMSE " << std::setprecision(4) << std::sqrt(se / test.size()) << "\n";

    // random initial factors, fixed seed so runs are reproducible
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> U((size_t)n_users * k), V((size_t)n_items * k);
    for (auto& x : U) x = dist(gen);
    for (auto& x : V) x = dist(gen);

    std::cout << "\nALS: k = " << k << ", lambda = " << lambda << "\n"
              << " iter   train RMSE   test RMSE   time (s)\n";
    for (int it = 1; it <= iters; it++) {
        t0 = Clock::now();
        als_update(R, V, U, k, lambda);    // fix V, solve for every user
        als_update(Rt, U, V, k, lambda);   // fix U, solve for every movie
        double t_iter = seconds_since(t0);
        std::cout << std::setw(5) << it << std::setw(13) << std::setprecision(4)
                  << rmse(train, U, V, k) << std::setw(12) << rmse(test, U, V, k)
                  << std::setw(11) << t_iter << "\n";
    }

    return 0;
}
