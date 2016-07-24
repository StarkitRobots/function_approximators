#include "regression_experiments/solver_factory.h"

#include "regression_experiments/gp_forest_solver.h"
#include "regression_experiments/gp_solver.h"
#include "regression_experiments/pwc_forest_solver.h"
#include "regression_experiments/pwl_forest_solver.h"

namespace regression_experiments
{

SolverFactory::SolverFactory()
{
  registerBuilder("gp_forest",
                  [](TiXmlNode * node)
                  {
                    Solver * solver = new GPForestSolver();
                    solver->from_xml(node);
                    return solver;
                  });
  registerBuilder("pwc_forest",
                  [](TiXmlNode * node)
                  {
                    (void)node;
                    return new PWCForestSolver();
                  });
  registerBuilder("pwl_forest",
                  [](TiXmlNode * node)
                  {
                    (void)node;
                    return new PWLForestSolver();
                  });
  registerBuilder("gp",
                  [](TiXmlNode * node)
                  {
                    Solver * solver = new GPSolver();
                    solver->from_xml(node);
                    return solver;
                  });
}

}
