// cfg.cpp
// -------
// Implementation of the simple key-value configuration reader.
//
// File format (one entry per line):
//   key = value
//
// Rules:
// - Lines may contain comments introduced by '#'. Everything after '#' is ignored.
// - Leading/trailing whitespace around keys/values is trimmed.
// - Empty lines are ignored.
// - Duplicate keys are allowed; the last occurrence wins.
//
// The parser is intentionally minimal (training code):
// - No nested sections, quoting, or escape sequences.
// - Values are stored as raw strings; typed getters convert on demand.

#include "cfg.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

// Trim leading and trailing whitespace.
static inline std::string trim(std::string s) {
    auto notSpace = [](unsigned char c){ return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

// Lowercase a string (ASCII). Used for tolerant boolean parsing.
static inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

void Cfg::load(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) throw std::runtime_error("Cfg: cannot open file: " + filename);

    // Start from a clean map for each load.
    kv_.clear();
    std::string line;
    int lineno = 0;

    // Read the file line-by-line and parse `key = value` entries.
    while (std::getline(ifs, line)) {
        ++lineno;

        // Strip comments.
        auto posHash = line.find('#');
        if (posHash != std::string::npos) line = line.substr(0, posHash);

        // Trim whitespace and skip empty lines.
        line = trim(line);
        if (line.empty()) continue;

        // Split `key = value` by the first '='.
        auto posEq = line.find('=');
        if (posEq == std::string::npos) {
            throw std::runtime_error("Cfg parse error at line " + std::to_string(lineno) + ": missing '='");
        }

        // Trim both key and value.
        std::string key = trim(line.substr(0, posEq));
        std::string val = trim(line.substr(posEq+1));
        if (key.empty()) {
            throw std::runtime_error("Cfg parse error at line " + std::to_string(lineno) + ": empty key");
        }

        // Store raw string value. If the key repeats, the last value wins.
        kv_[key] = val;
    }
}

// Check existence of a key.
bool Cfg::has(const std::string& key) const {
    return kv_.find(key) != kv_.end();
}

// Get string value (or return default).
std::string Cfg::getString(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    return (it == kv_.end()) ? def : it->second;
}

// Get integer value (or return default). Uses std::stoi.
int Cfg::getInt(const std::string& key, int def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    return std::stoi(it->second);
}

// Get floating-point value (or return default). Uses std::stod.
double Cfg::getDouble(const std::string& key, double def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    return std::stod(it->second);
}

// Get boolean value (or return default).
// Accepted true values:  1, true, yes, on
// Accepted false values: 0, false, no, off
// Any other value falls back to the provided default.
bool Cfg::getBool(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return def;
    std::string v = toLower(trim(it->second));
    if (v=="1"||v=="true"||v=="yes"||v=="on") return true;
    if (v=="0"||v=="false"||v=="no"||v=="off") return false;
    return def;
}
