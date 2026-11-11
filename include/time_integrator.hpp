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

    virtual ~TimeIntegrator() = default;
    virtual std::string name() const = 0;

    // state (in/out) - solution advanced by dt; rhs (in) - stage residual callback.
    virtual void step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs ) = 0;

  protected:
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

    std::string name() const override;

    void step( std::vector<Conservative_vector> &state, double dt, const RHSFunc &rhs ) override;

  private:
    std::vector<Conservative_vector> r_;
};


// Second-order two-stage Heun integration.
class RK2Integrator final : public TimeIntegrator
{
  public:
    using RHSFunc = TimeIntegrator::RHSFunc;

    std::string name() const override;

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

    std::string name() const override;

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

    std::string name() const override;

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
std::unique_ptr<TimeIntegrator> make_time_integrator( const std::string &name );
