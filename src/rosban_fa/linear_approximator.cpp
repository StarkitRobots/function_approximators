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

LinearApproximator::LinearApproximator(int input_dim,
                                       int output_dim,
                                       const Eigen::VectorXd & parameters,
                                       const Eigen::VectorXd & center)
  : LinearApproximator(input_dim, output_dim, parameters)
{
  if (input_dim != center.rows()) {
    std::ostringstream oss;
    oss << "LinearApproximator::LinearApproximator"
        << "(int,int,const Eigen::VectorXd &,const Eigen::VectorXd &): "
        << "center has " << parameters.rows() << " rows, expected: "
        << input_dim;
    throw std::runtime_error(oss.str());
  }
  bias = bias - coeffs * center;
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


Eigen::MatrixXd
LinearApproximator::getParametersSpace(const Eigen::MatrixXd & input_limits,
                                       const Eigen::MatrixXd & output_limits,
                                       bool narrow_slope) {
  // Storing number of dimensions
  int input_dims = input_limits.rows();
  int output_dims = output_limits.rows();
  int training_dims = output_dims * (input_dims + 1);
  
  Eigen::MatrixXd linear_parameters_space(training_dims,2);
  // Bias Limits
  linear_parameters_space.block(0,0,output_dims, 2) = output_limits;
  // Coeffs Limits
  // For each parameter, it might at most make the output vary from min to max in given space
  for (int output_dim = 0; output_dim < output_dims; output_dim++) {
    double output_amplitude = output_limits(output_dim,1) - output_limits(output_dim,0);
    for (int input_dim = 0; input_dim < input_dims; input_dim++) {
      int index = output_dim + output_dims * (1 + input_dim);
      double input_min = input_limits(input_dim,0);
      double input_max = input_limits(input_dim,1);
      // Avoiding numerical issues
      double input_amplitude = std::max(input_max - input_min,
                                        std::pow(10,-12));
      double max_coeff = output_amplitude / input_amplitude;
      // If narrow_slope is activated, then coefficients have to be combined
      // to make the output vary from min to max output
      if (narrow_slope) {
        max_coeff /= input_dims;
      }
      linear_parameters_space(index, 0) = -max_coeff;
      linear_parameters_space(index, 1) =  max_coeff;
    }
  }
  return linear_parameters_space;
}

Eigen::VectorXd
LinearApproximator::getDefaultParameters(const Eigen::MatrixXd & input_limits,
                                         const Eigen::MatrixXd & output_limits) {
  // Storing number of dimensions
  int input_dims = input_limits.rows();
  int output_dims = output_limits.rows();
  int training_dims = (input_dims+1) * output_dims;
  // Default slopes are 0
  Eigen::VectorXd initial_params = Eigen::VectorXd::Zero(training_dims);
  // Default bias is middle of output limits
  initial_params.segment(0,output_dims) = (output_limits.col(0) + output_limits.col(1)) / 2;
  
  return initial_params;
}

}
