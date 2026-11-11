#include "common_functions.hpp"


#include "cfg.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <cmath>
#include <stdexcept>


// Read key = value entries; the last value of a repeated key is retained.
void Cfg::load( const std::string &filename )
{
    std::ifstream ifs( filename );

    if ( !ifs )
    {
        throw std::runtime_error( "Cfg: cannot open file: " + filename );
    }

    // Replace the previous configuration; duplicate keys in this file retain the last value.
    kv_.clear();

    std::string line;
    int lineno = 0;

    while ( std::getline( ifs, line ) )
    {
        ++lineno;

        // Discard the # comment before splitting the key and value.
        auto pos_hash = line.find( '#' );

        if ( pos_hash != std::string::npos )
            line = line.substr( 0, pos_hash );

        line = trim_string( line );

        if ( line.empty() )
        {
            continue;
        }

        auto pos_eq = line.find( '=' );

        if ( pos_eq == std::string::npos )
        {
            throw std::runtime_error( "Cfg parse error at line " + std::to_string( lineno ) + ": missing '='" );
        }

        std::string key = trim_string( line.substr( 0, pos_eq ) );
        std::string val = trim_string( line.substr( pos_eq + 1 ) );

        if ( key.empty() )
        {
            throw std::runtime_error( "Cfg parse error at line " + std::to_string( lineno ) + ": empty key" );
        }

        kv_[key] = val;
    }
}


// Check whether a configuration key is present.
bool Cfg::has( const std::string &key ) const
{
    return kv_.find( key ) != kv_.end();
}


// Read a string value, or return def when the key is absent.
std::string Cfg::get_string( const std::string &key, const std::string &def ) const
{
    auto it = kv_.find( key );
    return ( it == kv_.end() ) ? def : it->second;
}


// Read an integer value; reject malformed or partially converted values.
int Cfg::get_int( const std::string &key, int def ) const
{
    auto it = kv_.find( key );

    if ( it == kv_.end() )
    {
        return def;
    }

    try
    {
        std::size_t used = 0;
        const int value = std::stoi( it->second, &used );

        if ( used != it->second.size() )
        {
            throw std::invalid_argument( "trailing characters" );
        }

        return value;
    }
    catch ( const std::exception & )
    {
        throw std::runtime_error( "Cfg: invalid integer for " + key + ": " + it->second );
    }
}


// Read a finite floating-point value; reject malformed or partially converted values.
double Cfg::get_double( const std::string &key, double def ) const
{
    auto it = kv_.find( key );

    if ( it == kv_.end() )
    {
        return def;
    }

    try
    {
        std::size_t used = 0;
        const double value = std::stod( it->second, &used );

        if ( used != it->second.size() || !std::isfinite( value ) )
        {
            throw std::invalid_argument( "invalid number" );
        }

        return value;
    }
    catch ( const std::exception & )
    {
        throw std::runtime_error( "Cfg: invalid finite number for " + key + ": " + it->second );
    }
}


// Read 1/true/yes/on or 0/false/no/off, ignoring letter case and surrounding whitespace.
bool Cfg::get_bool( const std::string &key, bool def ) const
{
    auto it = kv_.find( key );

    if ( it == kv_.end() )
    {
        return def;
    }

    std::string v = to_lowercase( trim_string( it->second ) );

    if ( v == "1" || v == "true" || v == "yes" || v == "on" )
    {
        return true;
    }

    if ( v == "0" || v == "false" || v == "no" || v == "off" )
    {
        return false;
    }

    throw std::runtime_error( "Cfg: invalid boolean for " + key + ": " + it->second );
}
