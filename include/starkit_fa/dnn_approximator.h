#pragma once

#include "starkit_fa/function_approximator.h"

#include "starkit_utils/serialization/json_serializable.h"

// tiny_dnn (and its dependencies) throw many warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif
#include "tiny_dnn/tiny_dnn.h"
#pragma GCC diagnostic pop

namespace starkit_fa
{
/// A function approximator based on multi-layer perceptron
///
/// Configuration of the DNN is ensured by DNNApproximator::Config
///
/// While it is based on tiny_dnn for training, it uses its own computation to
/// predict the value because prediction in tiny_dnn is not 'const'
class DNNApproximator : public FunctionApproximator
{
public:
  class Config : public starkit_utils::JsonSerializable, public starkit_utils::StreamSerializable
  {
  public:
    /// Format of output for the neural network
    enum OutputModel : int
    {
      Prediction = 0,   // The output of the NN is a simple estimation
      Distribution = 1  // The output of the NN is both, mean and variance for each variable
    };
    /// Activation function after each layer
    enum ActivationFunction : int
    {
      TanH = 0,
      Sigmoid = 1,
      Relu = 2
    };

    Config();

    // # Json Serialization
    virtual std::string getClassName() const override;
    virtual Json::Value toJson() const override;
    virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

    // # Stream Serialization
    virtual int getClassID() const override;
    virtual int writeInternal(std::ostream& out) const override;
    virtual int read(std::istream& in) override;

    /// Output model
    OutputModel output;

    /// Type of activation function after each layer
    ActivationFunction activation;

    /// Nb elements in hidden layer
    std::vector<int> layer_units;
  };

  typedef tiny_dnn::network<tiny_dnn::sequential> network;

  DNNApproximator();
  DNNApproximator(const network& nn, int input_dims, int output_dims, const Config& config);
  DNNApproximator(const DNNApproximator& other);

  virtual std::unique_ptr<FunctionApproximator> clone() const;

  int getOutputDim() const override;

  virtual void predict(const Eigen::VectorXd& input, Eigen::VectorXd& mean, Eigen::MatrixXd& covar) const override;

  virtual void gradient(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd& limits, Eigen::VectorXd& input, double& output) const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream& out) const override;
  virtual int read(std::istream& in) override;

  const network& getNetwork() const;

  /// Structure is as following:
  /// input_dim -> layer_units[0]
  /// ...
  /// layer_units[k] -> output_dim
  static network buildNN(int input_dim, int output_dim, const Config& config);

private:
  /// Synchronize hidden_layer_weights and final_layer_weights with current nn
  void updateWeightsFromNN();

  /// The neural network used to predict approximation
  network nn;

  /// Size of input
  int input_dim;

  /// Size of output
  int output_dim;

  /// The configuration of the DNN
  Config config;

  /// Weights of each layer of the DNN (coeffs, bias)
  std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> layer_weights;
};

}  // namespace starkit_fa
