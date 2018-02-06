#include "rhoban_fa/dnn_approximator.h"

#include "rhoban_utils/io_tools.h"

using namespace tiny_dnn;

namespace rhoban_fa
{

static std::string activation2Str(DNNApproximator::Config::ActivationFunction activation) {
  switch(activation) {
    case DNNApproximator::Config::ActivationFunction::TanH: return "TanH";
    case DNNApproximator::Config::ActivationFunction::Sigmoid: return "Sigmoid";
    case DNNApproximator::Config::ActivationFunction::Relu: return "Relu";
  }
  throw std::logic_error("activation2Str: unknown activation value");
}

static DNNApproximator::Config::ActivationFunction str2Activation(const std::string & str) {
  if (str == "TanH") return DNNApproximator::Config::ActivationFunction::TanH;
  if (str == "Sigmoid") return DNNApproximator::Config::ActivationFunction::Sigmoid;
  if (str == "Relu") return DNNApproximator::Config::ActivationFunction::Relu;
  throw std::logic_error("str2Activation: unknown str '" + str + "'");
}

static std::string outputModel2Str(DNNApproximator::Config::OutputModel model) {
  switch(model) {
    case DNNApproximator::Config::OutputModel::Prediction: return "Prediction";
    case DNNApproximator::Config::OutputModel::Distribution: return "Distribution";
  }
  throw std::logic_error("outputModel2Str: unknown activation value");
}

static DNNApproximator::Config::OutputModel str2OutputModel(const std::string & str) {
  if (str == "Prediction") return DNNApproximator::Config::OutputModel::Prediction;
  if (str == "Distribution") return DNNApproximator::Config::OutputModel::Distribution;
  throw std::logic_error("str2OutputModel: unknown str '" + str + "'");
}

DNNApproximator::Config::Config()
  : output(OutputModel::Prediction),
    activation(ActivationFunction::Relu),
    layer_units({1})
{
}

std::string DNNApproximator::Config::getClassName() const {
  return "DNNApproximator::Config";
}

Json::Value DNNApproximator::Config::toJson() const {
  Json::Value v;
  v["output"     ] = outputModel2Str(output);
  v["activation" ] = activation2Str(activation);
  v["layer_units"] = rhoban_utils::vector2Json(layer_units);
  return v;
}

void DNNApproximator::Config::fromJson(const Json::Value & v, const std::string & dir_name) {
  (void)dir_name;
  std::string output_str,activation_str;
  rhoban_utils::tryRead(v, "output", &output_str);
  rhoban_utils::tryRead(v, "activation", &activation_str);
  rhoban_utils::tryReadVector(v, "layer_units", &layer_units);
  if (output_str != "") output = str2OutputModel(output_str);
  if (activation_str != "") activation = str2Activation(activation_str);
}

int DNNApproximator::Config::getClassID() const {
  return 0;// No generic types for Config
}

int DNNApproximator::Config::writeInternal(std::ostream & out) const {
  int bytes_written = 0;
  bytes_written += rhoban_utils::write<int>(out, output);
  bytes_written += rhoban_utils::write<int>(out, activation);
  bytes_written += rhoban_utils::write<int>(out, layer_units.size());
  bytes_written += rhoban_utils::writeIntArray(out, layer_units.data(), layer_units.size());
  return bytes_written;
}

int DNNApproximator::Config::read(std::istream & in) {
  int bytes_read = 0;
  bytes_read += rhoban_utils::read<int>(in, (int*)&output);
  bytes_read += rhoban_utils::read<int>(in, (int*)&activation);
  int nb_layers;
  bytes_read += rhoban_utils::read<int>(in, &nb_layers);
  layer_units = std::vector<int>(nb_layers);
  rhoban_utils::readIntArray(in, layer_units.data(), nb_layers);
  return bytes_read;
}


DNNApproximator::DNNApproximator()
  : input_dim(0), output_dim(0)
{
}

DNNApproximator::DNNApproximator(const network & nn_, int input_dims, int output_dims,
                                 const Config & config)
  : nn(nn_), input_dim(input_dims), output_dim(output_dims), config(config)
{
  updateWeightsFromNN();
}

DNNApproximator::DNNApproximator(const DNNApproximator & other)
  : DNNApproximator(other.nn, other.input_dim, other.output_dim, other.config)
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
  Eigen::VectorXd tmp = input;
  for (size_t layer_idx = 0; layer_idx < layer_weights.size(); layer_idx++) {
    const Eigen::MatrixXd & coeffs = layer_weights[layer_idx].first;
    const Eigen::VectorXd & bias = layer_weights[layer_idx].second;
    tmp = coeffs * tmp + bias;
    // Last layer is not normalized
    if ( layer_idx != layer_weights.size() -1) {
      switch(config.activation) {
        case Config::ActivationFunction::TanH:
          for (int dim = 0; dim < tmp.rows(); dim++) {
            tmp(dim) = tanh(tmp(dim));
          }
          break;
        case Config::ActivationFunction::Sigmoid:
          for (int dim = 0; dim < tmp.rows(); dim++) {
            tmp(dim) = 1.0 / (1 + std::exp(-tmp(dim)));
          }
          break;
        case Config::ActivationFunction::Relu:
          for (int dim = 0; dim < tmp.rows(); dim++) {
            tmp(dim) = std::max(0.0,tmp(dim));
          }
          break;
      }
    }
  }
  switch(config.output) {
    case Config::OutputModel::Prediction:
      mean = tmp;
      break;
    case Config::OutputModel::Distribution:
      throw std::logic_error("DNNApproximator::predict: Distribution OutputModel not implemented");
    default:
      throw std::logic_error("DNNApproximator::predict: unknown OutputModel");
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

const DNNApproximator::network & DNNApproximator::getNetwork() const {
  return nn;
}

int DNNApproximator::getClassID() const {
  return FunctionApproximator::ID::DNNApproximator;
}

int DNNApproximator::writeInternal(std::ostream & out) const {
  int bytes_written = 0;
  bytes_written += rhoban_utils::write<int>(out, input_dim);
  bytes_written += rhoban_utils::write<int>(out, output_dim);
  bytes_written += config.writeInternal(out);
  for (size_t i = 0; i < nn.depth(); i++) {
    std::vector<const vec_t*> node_weights = nn[i]->weights();
    for (const vec_t * weights : node_weights){
      std::cout << "writing weights[" << i << "][.] (size " << weights->size() << ")" << std::endl;
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
  bytes_read += config.read(in);
  nn = buildNN(input_dim, output_dim, config);
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
  updateWeightsFromNN();
  return bytes_read;
}


DNNApproximator::network DNNApproximator::buildNN(int input_dim, int output_dim,
                                                  const Config & config) {
  network nn;
  int last_layer_dim = input_dim;
  for (int nb_units : config.layer_units) {
    switch(config.activation) {
      case Config::ActivationFunction::TanH:
        nn << fully_connected_layer<activation::tan_h>(last_layer_dim, nb_units);
        break;
      case Config::ActivationFunction::Sigmoid:
        nn << fully_connected_layer<activation::sigmoid>(last_layer_dim, nb_units);
        break;
      case Config::ActivationFunction::Relu:
        nn << fully_connected_layer<activation::relu>(last_layer_dim, nb_units);
        break;
      default:
        throw std::logic_error("DNNApproximator::buildNN: unknown activation function");
    }
    last_layer_dim = nb_units;
  }
  nn << fully_connected_layer<activation::identity>(last_layer_dim, output_dim);
  return nn;
}

static void fillEigenFromTiny(const vec_t & tiny, Eigen::VectorXd * eigen) {
  if (eigen->rows() != (int)tiny.size()) {
    throw std::logic_error("fillEigenFromTiny: size mismatch");
  }
  for (size_t idx = 0; idx < tiny.size(); idx++) {
    (*eigen)(idx) = tiny[idx];
  }
}

static void fillEigenFromTiny(const vec_t & tiny, Eigen::MatrixXd * eigen) {
  if (eigen->rows() * eigen->cols() != (int)tiny.size()) {
    throw std::logic_error("fillEigenFromTiny: size mismatch");
  }
  for (size_t idx = 0; idx < tiny.size(); idx++) {
    size_t row = idx % eigen->rows();
    size_t col = idx / eigen->rows();
    (*eigen)(row, col) = tiny[idx];
  }
}

void DNNApproximator::updateWeightsFromNN() {
  layer_weights.clear();
  int src_dim = input_dim;
  for (size_t layer_idx = 0; layer_idx < nn.depth(); layer_idx++) {
    bool output_layer = layer_idx == nn.depth() - 1;
    int dst_dim;
    if (output_layer) {
      dst_dim = output_dim;
    } else {
      dst_dim = config.layer_units[layer_idx];
    }
    Eigen::MatrixXd coeffs(dst_dim, src_dim);
    Eigen::VectorXd bias(dst_dim);
    fillEigenFromTiny(*(nn[layer_idx]->weights()[0]), &coeffs);
    fillEigenFromTiny(*(nn[layer_idx]->weights()[1]), &bias);
    layer_weights.push_back({coeffs,bias});
    // New input dim is last_layer dim
    src_dim = dst_dim;
  }
  
}

}
