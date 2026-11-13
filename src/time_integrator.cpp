#include "time_integrator.hpp"
#include "structures.hpp"
#include <stdexcept>


// Return the name of the time-integration scheme.
std::string EulerIntegrator::name() const
{
    return "euler";
}


// Advance the state by one time step using Forward Euler.
//
// state (in/out) - conserved variables in the ghosted cell field
// dt (in) - time increment for the current integration step
// rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
void EulerIntegrator::step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs )
{
    TimeIntegrator::ensure_size( r_, state.size() );

    // Evaluate the residual at the current state before the explicit update.
    rhs( state, r_ );

    for ( size_t i = 0; i < state.size(); ++i )
    {
        state[i] += dt * r_[i];
    }
}


// Return the name of the time-integration scheme.
std::string RK2Integrator::name() const
{
    return "rk2";
}


// Advance the state by one time step using two-stage Heun RK2.
//
// state (in/out) - conserved variables in the ghosted cell field
// dt (in) - time increment for the current integration step
// rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
//
void RK2Integrator::step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs )
{
    const size_t n = state.size();
    TimeIntegrator::ensure_size( u1_, n );
    TimeIntegrator::ensure_size( r_, n );

    // Evaluate the residual at the current state before the explicit update.
    rhs( state, r_ );

    for ( size_t i = 0; i < n; ++i )
    {
        u1_[i] = state[i] + dt * r_[i];
    }

    // Evaluate the next stage after its conservative state has been constructed.
    rhs( u1_, r_ );

    for ( size_t i = 0; i < n; ++i )
    {
        state[i] = 0.5 * state[i] + 0.5 * ( u1_[i] + dt * r_[i] );
    }
}


// Return the name of the time-integration scheme.
std::string SSPRK3Integrator::name() const
{
    return "ssprk3";
}


// Advance the state by one time step using three-stage Shu-Osher SSPRK3.
//
// state (in/out) - conserved variables in the ghosted cell field
// dt (in) - time increment for the current integration step
// rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
//
void SSPRK3Integrator::step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs )
{
    const size_t n = state.size();
    TimeIntegrator::ensure_size( u1_, n );
    TimeIntegrator::ensure_size( u2_, n );
    TimeIntegrator::ensure_size( r_, n );

    // Evaluate the residual at the current state before the explicit update.
    rhs( state, r_ );

    for ( size_t i = 0; i < n; ++i )
    {
        u1_[i] = state[i] + dt * r_[i];
    }

    // Evaluate the next stage after its conservative state has been constructed.
    rhs( u1_, r_ );

    for ( size_t i = 0; i < n; ++i )
    {
        u2_[i] = 0.75 * state[i] + 0.25 * ( u1_[i] + dt * r_[i] );
    }

    // Evaluate the final SSPRK3 stage before combining it with the original state.
    rhs( u2_, r_ );

    for ( size_t i = 0; i < n; ++i )
    {
        state[i] = ( 1.0 / 3.0 ) * state[i] + ( 2.0 / 3.0 ) * ( u2_[i] + dt * r_[i] );
    }
}


// Return the name of the time-integration scheme.
std::string RK4Integrator::name() const
{
    return "rk4";
}


// Advance the state by one time step using classical four-stage Runge-Kutta.
//
// state (in/out) - conserved variables in the ghosted cell field
// dt (in) - time increment for the current integration step
// rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
//
void RK4Integrator::step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs )
{
    const size_t n = state.size();
    TimeIntegrator::ensure_size( u0_, n );
    TimeIntegrator::ensure_size( stage_state_, n );
    TimeIntegrator::ensure_size( k1_, n );
    TimeIntegrator::ensure_size( k2_, n );
    TimeIntegrator::ensure_size( k3_, n );
    TimeIntegrator::ensure_size( k4_, n );

    for ( size_t i = 0; i < n; ++i )
    {
        u0_[i] = state[i];
    }

    // Evaluate k1 at the initial state; k2 and k3 use half steps, and k4 uses a full step.
    rhs( state, k1_ );

    for ( size_t i = 0; i < n; ++i )
    {
        stage_state_[i] = u0_[i] + 0.5 * dt * k1_[i];
    }

    rhs( stage_state_, k2_ );

    for ( size_t i = 0; i < n; ++i )
    {
        stage_state_[i] = u0_[i] + 0.5 * dt * k2_[i];
    }

    rhs( stage_state_, k3_ );

    for ( size_t i = 0; i < n; ++i )
    {
        stage_state_[i] = u0_[i] + dt * k3_[i];
    }

    rhs( stage_state_, k4_ );

    for ( size_t i = 0; i < n; ++i )
    {
        // Combine the four stage derivatives with weights 1/6, 1/3, 1/3 and 1/6.
        state[i] = u0_[i] + dt * ( k1_[i] + 2.0 * k2_[i] + 2.0 * k3_[i] + k4_[i] ) / 6.0;
    }
}


// Create the selected time integrator: euler, rk2, ssprk3 or rk4.
//
// name (in) - requested numerical-method name
std::unique_ptr<TimeIntegrator> make_time_integrator( const std::string &name )
{
    if ( name == "euler" )
    {
        return std::make_unique<EulerIntegrator>();
    }

    if ( name == "rk2" )
    {
        return std::make_unique<RK2Integrator>();
    }

    if ( name == "ssprk3" )
    {
        return std::make_unique<SSPRK3Integrator>();
    }

    if ( name == "rk4" )
    {
        return std::make_unique<RK4Integrator>();
    }

    throw std::runtime_error( "Unknown timeIntegrator: " + name );
}
