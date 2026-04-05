#pragma once

// cfg.hpp
// Simple key-value configuration reader.
//
// - The config file is parsed into a string->string map (kv_).
// - Typed getters (int/double/bool/string) convert from the stored string.
// - Call load(filename) once at program start, then query values via getters.
//
// Design goal: keep configuration handling minimal and easy to extend.

#include <string>
#include <unordered_map>

// Cfg: in-memory view of a config file.
//
// Keys are stored exactly as written in the cfg file (case-sensitive).
// Recommended convention for hierarchical options: use dotted keys, e.g.
//   reconstruction.scheme = weno5
//   bc.left = reflective
//
// Typical usage:
//   Cfg cfg;
//   cfg.load("cases/case1d.cfg");
//   const int nx = cfg.getInt("nx", 200);
//
// This header declares the interface; see cfg.cpp for parsing details.
class Cfg {
public:
    // Parse a cfg file and fill the internal key-value map.
    // Existing keys are overwritten if the file contains duplicates.
    void load(const std::string& filename);

    // Return true if the key exists in the configuration map.
    bool has(const std::string& key) const;

    // Get a string value.
    // If the key does not exist, returns `def`.
    std::string getString(const std::string& key, const std::string& def="") const;
    // Get an integer value (converted from the stored string).
    // If the key does not exist, returns `def`.
    int         getInt   (const std::string& key, int def=0) const;
    // Get a floating-point value (converted from the stored string).
    // If the key does not exist, returns `def`.
    double      getDouble(const std::string& key, double def=0.0) const;
    // Get a boolean value.
    // Recommended accepted values in the cfg file: true/false, 1/0 (see cfg.cpp).
    // If the key does not exist, returns `def`.
    bool        getBool  (const std::string& key, bool def=false) const;

private:
    // Internal storage: raw string values as read from the file.
    // Typed getters convert from this representation.
    std::unordered_map<std::string, std::string> kv_;
};
