// cc_vis.cpp
//
// Standalone visual aid for studying connected-components algorithms on a
// 2D 8x8 grid.  Single file, no NanoVDB / CUDA / OpenVDB dependencies.
//
// Pass 1: initialize parents[] from one of four alive-cell patterns and
//         print the resulting label grid.
// Pass 2 (this version): interactive loop that applies Propagate / SV-Hook /
//         Compress one step at a time, re-printing the grid after each step.
//
// Build:
//   g++ -std=c++17 -O2 cc_vis.cpp -o cc_vis
//
// Run:
//   ./cc_vis [pattern]
//     pattern (optional) is one of: full | snake | band | net
//     if omitted, an interactive prompt is shown.
//
// Labels: row-major linear, label(i,j) = i*N + j.
//
// Operations (4-connectivity, face neighbors only):
//   propagate:  parents_new[v] = min(parents[v], min{parents[n] : n in N(v)})
//   hook (SV):  Phase A copies parents -> parents_new; Phase B, for each v,
//               if min{parents[n] : n in N(v)} < parents[v], it lowers
//               parents_new[parents[v]] (the *root* slot) toward that min.
//   compress:   parents_new[v] = parents[parents[v]]   (pointer jumping)

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

namespace {

constexpr int N = 8;
constexpr int BARRIER = -1;

enum class Pattern { Full, Snake, Band, Net, Invalid };

struct Grid {
    // NB: using raw 2D C arrays rather than nested std::array.  GCC 13 at -O1+
    // was eliding writes from initLabels() when these were nested std::array
    // (writes did happen with -O0, with UBSAN, or with __attribute__((noinline))
    // -- looks like an optimizer interaction; raw arrays sidestep it cleanly).
    std::uint8_t alive[N][N]{};
    int          parents[N][N]{};
    int          parents_new[N][N]{};

    static int label(int i, int j) { return i * N + j; }
    static std::pair<int, int> labelToCoord(int lab) { return {lab / N, lab % N}; }

    // Iterate over the 4 face-neighbors (i±1, j) and (i, j±1) that are in bounds.
    template <typename F>
    void forEachNeighbor(int i, int j, F&& f) const {
        if (i > 0)     f(i - 1, j);
        if (i < N - 1) f(i + 1, j);
        if (j > 0)     f(i, j - 1);
        if (j < N - 1) f(i, j + 1);
    }

    void initLabels() {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                parents[i][j] = alive[i][j] ? label(i, j) : BARRIER;
    }

    int countAlive() const {
        int c = 0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (alive[i][j]) ++c;
        return c;
    }

    void print() const {
        std::cout << "       ";
        for (int j = 0; j < N; ++j) std::cout << "  c" << j << " ";
        std::cout << "\n";

        for (int i = 0; i < N; ++i) {
            std::cout << "row " << i << "  ";
            for (int j = 0; j < N; ++j) {
                if (alive[i][j]) {
                    std::cout << "[" << std::setw(3) << parents[i][j] << "]";
                } else {
                    std::cout << "[   ]";
                }
            }
            std::cout << "\n";
        }
    }

    // ---- Operations ---------------------------------------------------------

    // Propagate:  parents_new[v] = min(parents[v], min{parents[n] : n alive in N(v)})
    //   Writes are partitioned (each thread writes own slot) -- no atomics.
    void propagate() {
        std::memcpy(parents_new, parents, sizeof(parents));  // barriers carry through unchanged
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (!alive[i][j]) continue;
                int m = parents[i][j];
                forEachNeighbor(i, j, [&](int ni, int nj) {
                    if (alive[ni][nj]) m = std::min(m, parents[ni][nj]);
                });
                parents_new[i][j] = m;
            }
        }
        std::swap(parents, parents_new);
    }

    // SV-Hook (Shiloach-Vishkin root hook), double-buffered:
    //   Phase A: parents_new = parents   (each thread writes own slot)
    //   Barrier
    //   Phase B: for each alive v:
    //              m = min{parents[n] : n alive in N(v)}
    //              if m < parents[v]:
    //                  atomicMin(&parents_new[parents[v]], m)   // root slot
    //   In single-threaded CPU code "atomicMin" is just std::min.
    void hook() {
        std::memcpy(parents_new, parents, sizeof(parents));  // Phase A
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (!alive[i][j]) continue;
                int pv = parents[i][j];
                int neighbor_min = std::numeric_limits<int>::max();
                forEachNeighbor(i, j, [&](int ni, int nj) {
                    if (alive[ni][nj]) {
                        neighbor_min = std::min(neighbor_min, parents[ni][nj]);
                    }
                });
                if (neighbor_min < pv) {
                    auto [pi, pj] = labelToCoord(pv);
                    parents_new[pi][pj] = std::min(parents_new[pi][pj], neighbor_min);
                }
            }
        }
        std::swap(parents, parents_new);
    }

    // Compress: parents_new[v] = parents[parents[v]]   (pointer jumping)
    //   Writes partitioned (own slot), no atomics, no init pass needed,
    //   no within-iteration barrier.
    void compress() {
        std::memcpy(parents_new, parents, sizeof(parents));  // (only alive cells get overwritten below)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (!alive[i][j]) continue;
                int p = parents[i][j];
                auto [pi, pj] = labelToCoord(p);
                parents_new[i][j] = parents[pi][pj];
            }
        }
        std::swap(parents, parents_new);
    }
};

// ---- Pattern application ----------------------------------------------------

void applyFull(Grid& g) {
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) g.alive[i][j] = 1;
}

// U-shaped meandering snake:
//   even rows: fully alive
//   odd rows (i % 4 == 1): only rightmost cell  (transitions from row above)
//   odd rows (i % 4 == 3): only leftmost cell
void applySnake(Grid& g) {
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            for (int j = 0; j < N; ++j) g.alive[i][j] = true;
        } else if (i % 4 == 1) {
            g.alive[i][N - 1] = true;
        } else { // i % 4 == 3
            g.alive[i][0] = true;
        }
    }
}

void applyBand(Grid& g) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            g.alive[i][j] = (i + j >= 3) && (i + j <= 11);
}

void applyNet(Grid& g) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            g.alive[i][j] = (i % 2 == 0) || (j % 2 == 0);
}

void applyPattern(Pattern p, Grid& g) {
    std::memset(g.alive, 0, sizeof(g.alive));
    switch (p) {
        case Pattern::Full:  applyFull(g);  break;
        case Pattern::Snake: applySnake(g); break;
        case Pattern::Band:  applyBand(g);  break;
        case Pattern::Net:   applyNet(g);   break;
        default: break;
    }
}

// ---- Pattern name <-> enum --------------------------------------------------

const char* patternName(Pattern p) {
    switch (p) {
        case Pattern::Full:  return "FULL";
        case Pattern::Snake: return "SNAKE";
        case Pattern::Band:  return "BAND";
        case Pattern::Net:   return "NET";
        default:             return "INVALID";
    }
}

Pattern parsePattern(const std::string& s) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "1" || t == "full")  return Pattern::Full;
    if (t == "2" || t == "snake") return Pattern::Snake;
    if (t == "3" || t == "band")  return Pattern::Band;
    if (t == "4" || t == "net")   return Pattern::Net;
    return Pattern::Invalid;
}

Pattern promptForPattern() {
    while (true) {
        std::cout << "Choose initialization pattern:\n"
                  << "  1) full   - All 64 cells alive\n"
                  << "  2) snake  - U-shaped meandering snake\n"
                  << "  3) band   - Diagonal band:  3 <= i+j <= 11\n"
                  << "  4) net    - Sparse net:     (i%2==0) OR (j%2==0)\n"
                  << "Enter choice [1-4 or name]: ";
        std::string line;
        if (!std::getline(std::cin, line)) {
            std::cerr << "\nEOF on input.\n";
            std::exit(1);
        }
        Pattern p = parsePattern(line);
        if (p != Pattern::Invalid) return p;
        std::cout << "  '" << line << "' not recognized. Try again.\n\n";
    }
}

// ---- Interactive operation loop --------------------------------------------

char lowerFirst(const std::string& s) {
    if (s.empty()) return '\0';
    return static_cast<char>(std::tolower(static_cast<unsigned char>(s[0])));
}

void runOperationLoop(Grid& g, Pattern pattern) {
    int nProp = 0, nHook = 0, nComp = 0;

    while (true) {
        std::cout << "\nOperations  (counts: P=" << nProp
                  << "  H=" << nHook
                  << "  C=" << nComp << ")\n"
                  << "  p) propagate  - one-hop label propagation (vertex-min)\n"
                  << "  h) hook       - SV-style root-hook (atomicMin -> root slot)\n"
                  << "  c) compress   - pointer jumping (parents[parents[v]])\n"
                  << "  r) reset      - re-init parents (same pattern)\n"
                  << "  q) quit\n"
                  << "Enter choice: ";

        std::string line;
        if (!std::getline(std::cin, line)) {
            std::cout << "\n";
            return;
        }
        const char c = lowerFirst(line);

        if (c == 'q') return;

        if (c == 'r') {
            g.initLabels();
            nProp = nHook = nComp = 0;
            std::cout << "\n=== RESET to initial labels (" << patternName(pattern) << ") ===\n\n";
            g.print();
            continue;
        }

        if (c == 'p' || c == 'h' || c == 'c') {
            std::string label_str;
            if (c == 'p') { g.propagate(); ++nProp; label_str = "PROPAGATE #" + std::to_string(nProp); }
            if (c == 'h') { g.hook();      ++nHook; label_str = "HOOK #"      + std::to_string(nHook); }
            if (c == 'c') { g.compress();  ++nComp; label_str = "COMPRESS #"  + std::to_string(nComp); }
            std::cout << "\n=== After " << label_str << " ===\n\n";
            g.print();
            continue;
        }

        std::cout << "  '" << line << "' not recognized.\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    Pattern pattern = Pattern::Invalid;

    if (argc >= 2) {
        pattern = parsePattern(argv[1]);
        if (pattern == Pattern::Invalid) {
            std::cerr << "Unknown pattern '" << argv[1]
                      << "'. Falling back to prompt.\n\n";
        }
    }
    if (pattern == Pattern::Invalid) pattern = promptForPattern();

    Grid g;
    applyPattern(pattern, g);
    g.initLabels();

    std::cout << "\n=== Initialization pattern: " << patternName(pattern) << " ===\n\n";
    g.print();
    const int alive = g.countAlive();
    std::cout << "\nAlive: " << alive
              << " cells.  Barriers: " << (N * N - alive) << " cells.\n";

    runOperationLoop(g, pattern);
    return 0;
}
