#pragma once


#include "structures.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>


// Advance dU/dt = RHS(U); the RHS callback may refresh stage ghost cells.
class TimeIntegrator
{
  public:
    // RHS callback: first argument (in/out) is the stage state, second argument (out) is its residual.
    // Both arrays have the same ghosted layout; the callback may update stage ghost cells.
    using RHSFunc = std::function<void( std::vector<Conservative_vector> &, std::vector<Conservative_vector> & )>;

    // Release owned resources during object destruction.
    virtual ~TimeIntegrator() = default;
    // Return the name of the time-integration scheme.
    virtual std::string name() const = 0;

    // Advance the state by one time step using this integrator and the stage residual callback.
    //
    // state (in/out) - conserved variables in the ghosted cell field
    // dt (in) - time increment for the current integration step
    // rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
    virtual void step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs ) = 0;

  protected:
    // Resize a reusable stage buffer when its current size differs from the requested size.
    //
    // buf (in/out) - reusable integration buffer whose size is adjusted
    // n (in) - requested number of buffer elements
    static void ensure_size( std::vector<Conservative_vector> &buf, size_t n )
    {
        if ( buf.size() != n )
        {
            buf.resize( n );
        }
    }
};


// First-order forward Euler integration.
class EulerIntegrator final : public TimeIntegrator
{
  public:
    using RHSFunc = TimeIntegrator::RHSFunc;

    // Return the name of the time-integration scheme.
    std::string name() const override;

    // Advance the state by one time step using this integrator and the stage residual callback.
    //
    // state (in/out) - conserved variables in the ghosted cell field
    // dt (in) - time increment for the current integration step
    // rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
    void step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs ) override;

  private:
    std::vector<Conservative_vector> r_;
};


// Second-order two-stage Heun integration.
class RK2Integrator final : public TimeIntegrator
{
  public:
    using RHSFunc = TimeIntegrator::RHSFunc;

    // Return the name of the time-integration scheme.
    std::string name() const override;

    // Advance the state by one time step using this integrator and the stage residual callback.
    //
    // state (in/out) - conserved variables in the ghosted cell field
    // dt (in) - time increment for the current integration step
    // rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
    void step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs ) override;

  private:
    std::vector<Conservative_vector> u1_;
    std::vector<Conservative_vector> r_;
};


// Third-order three-stage Shu-Osher SSP integration.
class SSPRK3Integrator final : public TimeIntegrator
{
  public:
    using RHSFunc = TimeIntegrator::RHSFunc;

    // Return the name of the time-integration scheme.
    std::string name() const override;

    // Advance the state by one time step using this integrator and the stage residual callback.
    //
    // state (in/out) - conserved variables in the ghosted cell field
    // dt (in) - time increment for the current integration step
    // rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
    void step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs ) override;

  private:
    std::vector<Conservative_vector> u1_;
    std::vector<Conservative_vector> u2_;
    std::vector<Conservative_vector> r_;
};


// Fourth-order classical four-stage Runge-Kutta integration.
class RK4Integrator final : public TimeIntegrator
{
  public:
    using RHSFunc = TimeIntegrator::RHSFunc;

    // Return the name of the time-integration scheme.
    std::string name() const override;

    // Advance the state by one time step using this integrator and the stage residual callback.
    //
    // state (in/out) - conserved variables in the ghosted cell field
    // dt (in) - time increment for the current integration step
    // rhs (in) - stage residual callback; may refresh stage ghost cells before writing the residual
    void step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs ) override;

  private:
    std::vector<Conservative_vector> u0_;
    std::vector<Conservative_vector> u1_;
    std::vector<Conservative_vector> u2_;
    std::vector<Conservative_vector> u3_;
    std::vector<Conservative_vector> k1_;
    std::vector<Conservative_vector> k2_;
    std::vector<Conservative_vector> k3_;
    std::vector<Conservative_vector> k4_;
};


// Create the selected time integrator: euler, rk2, ssprk3 or rk4.
//
// name (in) - requested numerical-method name
std::unique_ptr<TimeIntegrator> make_time_integrator( const std::string &name );
