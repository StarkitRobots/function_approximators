#include "rosban_fa/factory.h"

#include "rosban_fa/gp_forest.h"
#include "rosban_fa/gp.h"
#include "rosban_fa/pwc_forest.h"
#include "rosban_fa/pwl_forest.h"

namespace rosban_fa
{

Factory::Factory()
{
  registerBuilder("gp_forest",
                  [](TiXmlNode * node)
                  {
                    FunctionApproximator * solver = new GPForest();
                    solver->from_xml(node);
                    return solver;
                  });
  registerBuilder("pwc_forest",
                  [](TiXmlNode * node)
                  {
                    (void)node;
                    return new PWCForest();
                  });
  registerBuilder("pwl_forest",
                  [](TiXmlNode * node)
                  {
                    (void)node;
                    return new PWLForest();
                  });
  registerBuilder("gp",
                  [](TiXmlNode * node)
                  {
                    FunctionApproximator * solver = new GP();
                    solver->from_xml(node);
                    return solver;
                  });
}

}
