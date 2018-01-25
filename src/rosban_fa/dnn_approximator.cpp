#include "rosban_fa/dnn_approximator.h"

#include "rhoban_utils/io_tools.h"

using namespace tiny_dnn;

namespace rosban_fa
{

DNNApproximator::DNNApproximator()
  : input_dim(0), output_dim(0), nb_units(1)
{
}

DNNApproximator::DNNApproximator(const network & nn_, int input_dims, int output_dims, int nb_units)
  : nn(nn_), input_dim(input_dims), output_dim(output_dims), nb_units(nb_units)
{
}

DNNApproximator::DNNApproximator(const DNNApproximator & other)
  : DNNApproximator(other.nn, other.input_dim, other.output_dim, other.nb_units)
{
}

std::unique_ptr<FunctionApproximator> DNNApproximator::clone() const {
  return std::unique_ptr<FunctionApproximator>(new DNNApproximator(*this));
}

int DNNApproximator::getOutputDim() const {
  return output_dim;
}

void DNNApproximator::predict(const Eigen::VectorXd & input,
                              Eigen::VectorXd & mean,
                              Eigen::MatrixXd & covar) const {
  (void) covar;// Currently not filling covariance matrix
  network copy(nn);//TODO: should be fixed, but require strong modifications of tiny-dnn
  // Convert input to tiny format
  vec_t tiny_input;
  for (int row = 0; row < input.rows(); row++) {
    tiny_input.push_back(input(row));
  }
  vec_t res = copy.predict(tiny_input);
  mean = Eigen::VectorXd(output_dim);
  for (int d = 0; d < output_dim; d++) {
    mean(d) = res[d];
  }
}

void DNNApproximator::gradient(const Eigen::VectorXd & input,
                               Eigen::VectorXd & gradient) const
{
  (void) input; (void) gradient;
  throw std::logic_error("DNNApproximator::gradient: not implemented");
}

void DNNApproximator::getMaximum(const Eigen::MatrixXd & limits,
                                 Eigen::VectorXd & input,
                                 double & output) const
{
  (void) input; (void) output; (void) limits;
  throw std::logic_error("DNNApproximator::getMaximum: not implemented");
}

int DNNApproximator::getClassID() const {
  return FunctionApproximator::ID::DNNApproximator;
}

int DNNApproximator::writeInternal(std::ostream & out) const {
  int bytes_written = 0;
  bytes_written += rhoban_utils::write<int>(out, input_dim);
  bytes_written += rhoban_utils::write<int>(out, output_dim);
  bytes_written += rhoban_utils::write<int>(out, nb_units);
  for (size_t i = 0; i < nn.depth(); i++) {
    std::vector<const vec_t*> node_weights = nn[i]->weights();
    for (const vec_t * weights : node_weights){
      std::vector<double> weights_double(weights->size());
      for (size_t idx = 0; idx < weights->size(); idx++) {
        weights_double[idx] = (*weights)[idx];
      }
      bytes_written += rhoban_utils::writeDoubleArray(out, weights_double.data(), weights->size());
    }
  }
  return bytes_written;
}

int DNNApproximator::read(std::istream & in) {
  int bytes_read = 0;
  bytes_read += rhoban_utils::read<int>(in, &input_dim);
  bytes_read += rhoban_utils::read<int>(in, &output_dim);
  bytes_read += rhoban_utils::read<int>(in, &nb_units);
  nn = buildNN(input_dim, output_dim, nb_units);
  for (size_t i = 0; i < nn.depth(); i++) {
    std::vector<vec_t*> node_weights = nn[i]->weights();
    for (vec_t * weights : node_weights){
      std::vector<double> values(weights->size());
      bytes_read += rhoban_utils::readDoubleArray(in, values.data(), weights->size());
      for (size_t idx = 0; idx < weights->size(); idx++) {
        (*weights)[idx] = values[idx];
      }
    }
  }
  return bytes_read;
}


DNNApproximator::network DNNApproximator::buildNN(int input_dim, int output_dim, int nb_units) {
  network nn;
  nn << fully_connected_layer<activation::tan_h>(input_dim, nb_units)
     << fully_connected_layer<activation::identity>(nb_units, output_dim);
  return nn;
}

}
