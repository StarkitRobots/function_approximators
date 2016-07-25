#include "rosban_fa/trainer.h"

#include <sstream>
#include <stdexcept>

namespace rosban_fa
{

Trainer::~Trainer() {}


void Trainer::checkConsistency(const Eigen::MatrixXd & inputs,
                               const Eigen::MatrixXd & observations,
                               const Eigen::MatrixXd & limits) const
{
  if (inputs.cols() != observations.rows()) {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: inconsistent number of samples: "
        << "inputs.cols() != observations.rows() "
        << "(" << inputs.cols() << " != " << observations.rows() << ")";
    throw std::logic_error(oss.str());
  }
  if (limits.rows() != inputs.rows()) {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: inconsistent dimension for input: "
        << "inputs.rows() != limits.rows() "
        << "(" << inputs.rows() << " != " << limits.rows() << ")";
    throw std::logic_error(oss.str());
  }
  if (limits.cols() != 2) {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: invalid dimensions for limits: "
        << "expecting 2 columns and received " << limits.rows();
    throw std::logic_error(oss.str());
  }
}

}
