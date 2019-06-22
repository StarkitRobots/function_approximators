#include "starkit_fa/constant_approximator.h"

#include "starkit_utils/io_tools.h"

namespace starkit_fa
{
ConstantApproximator::ConstantApproximator()
{
}

ConstantApproximator::ConstantApproximator(const Eigen::VectorXd& average_) : average(average_)
{
}

ConstantApproximator::~ConstantApproximator()
{
}

std::unique_ptr<FunctionApproximator> ConstantApproximator::clone() const
{
  return std::unique_ptr<FunctionApproximator>(new ConstantApproximator(average));
}

const Eigen::VectorXd& ConstantApproximator::getValue() const
{
  return average;
}

int ConstantApproximator::getOutputDim() const
{
  return average.rows();
}

void ConstantApproximator::predict(const Eigen::VectorXd& input, Eigen::VectorXd& mean, Eigen::MatrixXd& covar) const
{
  (void)input;
  mean = average;
  covar = Eigen::MatrixXd::Zero(getOutputDim(), getOutputDim());
}

void ConstantApproximator::gradient(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) const
{
  check1DOutput("getMaximum");
  gradient = Eigen::VectorXd::Zero(input.rows());
}

void ConstantApproximator::getMaximum(const Eigen::MatrixXd& limits, Eigen::VectorXd& input, double& output) const
{
  check1DOutput("getMaximum");
  input = (limits.col(0) + limits.col(1)) / 2.0;
  output = average(0);
}

int ConstantApproximator::getClassID() const
{
  return Constant;
}

int ConstantApproximator::writeInternal(std::ostream& out) const
{
  int bytes_written = 0;
  int dim = getOutputDim();
  bytes_written += starkit_utils::write<int>(out, dim);
  bytes_written += starkit_utils::writeArray<double>(out, dim, average.data());
  return bytes_written;
}

int ConstantApproximator::read(std::istream& in)
{
  int bytes_read = 0;
  int output_dim;
  bytes_read += starkit_utils::read<int>(in, &output_dim);
  average = Eigen::VectorXd(output_dim);
  bytes_read += starkit_utils::readDoubleArray(in, average.data(), output_dim);
  return bytes_read;
}

std::string ConstantApproximator::toString() const
{
  std::ostringstream oss;
  oss << "(ConstantApproximator| avg: " << average.transpose() << " )";
  return oss.str();
}

}  // namespace starkit_fa
