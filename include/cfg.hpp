#pragma once


#include <string>
#include <unordered_map>


// Store case-sensitive configuration keys and their string values; typed getters reject invalid values.
class Cfg
{
  public:
    // filename (in) - key = value file; # begins a line comment.
    void load( const std::string &filename );

    bool has( const std::string &key ) const;

    // Missing keys return def; malformed present values in typed getters throw exceptions.
    std::string get_string( const std::string &key, const std::string &def = "" ) const;

    int get_int( const std::string &key, int def = 0 ) const;

    double get_double( const std::string &key, double def = 0.0 ) const;

    bool get_bool( const std::string &key, bool def = false ) const;

  private:
    std::unordered_map<std::string, std::string> kv_;
};
