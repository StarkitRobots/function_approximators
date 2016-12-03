#include "rosban_fa/linear_approximator.h"

#include "rosban_utils/io_tools.h"

namespace rosban_fa
{

LinearApproximator::LinearApproximator() {}

LinearApproximator::LinearApproximator(const Eigen::VectorXd & bias_,
                                       const Eigen::MatrixXd & coeffs_)
  : bias(bias_), coeffs(coeffs_)
{
}

LinearApproximator::LinearApproximator(int input_dim,
                                       int output_dim,
                                       const Eigen::VectorXd & parameters)
{
  int expected_rows = (input_dim + 1) * output_dim;
  if (parameters.rows() != expected_rows) {
    std::ostringstream oss;
    oss << "LinearApproximator::LinearApproximator(int,int,const Eigen::VectorXd &): "
        << "parameters has " << parameters.rows() << " rows, expected: "
        << expected_rows;
    throw std::runtime_error(oss.str());
  }
  bias = parameters.segment(0,output_dim);
  coeffs = Eigen::MatrixXd(output_dim, input_dim);
  for (int col = 0; col < input_dim; col++) {
    coeffs.col(col) = parameters.segment((col + 1) * output_dim, output_dim);
  }
}

LinearApproximator::~LinearApproximator()
{
}

int LinearApproximator::getOutputDim() const
{
  return bias.rows();
}

void LinearApproximator::predict(const Eigen::VectorXd & input,
                                 Eigen::VectorXd & mean,
                                 Eigen::MatrixXd & covar) const
{
  (void) input;
  mean = bias + coeffs * input;
  covar = Eigen::MatrixXd::Zero(getOutputDim(),getOutputDim());
}

void LinearApproximator::gradient(const Eigen::VectorXd & input,
                                  Eigen::VectorXd & gradient) const
{
  (void) input;
  check1DOutput("getMaximum");
  gradient = coeffs.row(0);
}

void LinearApproximator::getMaximum(const Eigen::MatrixXd & limits,
                                    Eigen::VectorXd & input,
                                    double & output) const
{
  (void) limits;
  (void) input;
  (void) output;
  check1DOutput("getMaximum");
  throw std::logic_error("LinearApproximator::getMaximum is not implemented");
}

int LinearApproximator::getClassID() const
{
  return Linear;
}

int LinearApproximator::writeInternal(std::ostream & out) const
{
  int bytes_written = 0;
  int output_dim = getOutputDim();
  int input_dim  = coeffs.cols();
  bytes_written += rosban_utils::write<int>(out, output_dim);
  bytes_written += rosban_utils::write<int>(out, input_dim);
  bytes_written += rosban_utils::writeArray<double>(out, output_dim, bias.data());
  bytes_written += rosban_utils::writeArray<double>(out, input_dim * output_dim, coeffs.data());
  return bytes_written;
}

int LinearApproximator::read(std::istream & in)
{
  int bytes_read = 0;
  int output_dim, input_dim;
  bytes_read += rosban_utils::read<int>(in, &output_dim);
  bytes_read += rosban_utils::read<int>(in, &input_dim);
  bias = Eigen::VectorXd(output_dim);
  coeffs = Eigen::MatrixXd(output_dim, input_dim);
  bytes_read += rosban_utils::readDoubleArray(in, bias.data()  , output_dim);
  bytes_read += rosban_utils::readDoubleArray(in, coeffs.data(), output_dim * input_dim);
  return bytes_read;
}

std::string LinearApproximator::toString() const {
  std::ostringstream oss;
  oss << "(LinearApproximator| bias: " << bias.transpose() << " | coeffs: ";
  for (int row = 0; row < coeffs.rows(); row++) {
    oss << coeffs.row(row);
    if (row != coeffs.rows() - 1) oss << "; ";
  }
  oss << ")";
  return oss.str();
}

}
