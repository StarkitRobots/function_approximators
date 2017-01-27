#pragma once

#include "rosban_fa/function_approximator.h"

namespace rosban_fa
{

/// This class implements approximation of a function with a linear model
/// prediction = A*(X-O) + B
class LinearApproximator : public FunctionApproximator
{
public:

  LinearApproximator();
  LinearApproximator(const Eigen::VectorXd & bias,
                     const Eigen::MatrixXd & coeffs);

  /// Parameters are ordered in the following way:
  /// bias, then coeffs.col(0), then coeffs.col(1), ...
  /// Throws a runtime_error if the number of parameters is not appropriate
  LinearApproximator(int input_dim, int output_dim,
                     const Eigen::VectorXd & parameters);

  /// In this constructor, an additional parameter specifies the center position
  /// used for expressing the parameters.
  /// Y = bias + A * (X-center)
  /// bias is directly corrected using coeffients from A
  LinearApproximator(int input_dim, int output_dim,
                     const Eigen::VectorXd & parameters,
                     const Eigen::VectorXd & center);

  virtual ~LinearApproximator();

  virtual std::unique_ptr<FunctionApproximator> clone() const override;

  Eigen::VectorXd getBias() const;
  Eigen::VectorXd getBias(const Eigen::VectorXd & center) const;
  const Eigen::MatrixXd & getCoeffs() const;
  /// Output the coeff as a vector (M_0,0;M_1,0;...;M_0,1;M1,1;...)
  Eigen::VectorXd getCoeffsAsVector() const;

  virtual int getOutputDim() const override;

  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) const override;

  virtual void gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;

  virtual std::string toString() const override;

  /// Return the parameters limits for the given limits.
  /// Those parameters are based on use of the constructor which include the
  /// center of the input space.
  /// Provided limits are chosen to ensure the following properties:
  /// - When narrow_slope is true:
  ///   - It is possible that inside input_limits, min and max of output limits
  ///     are predicted, but then both values necessarily appears on 'corners' of
  ///     the hyperrectangle
  /// - When narrow_slope is false:
  ///   - Same as when narrow_slope is true except extremum values can appear on
  ///     opposite 'borders' (i.e. only 1 dimension has changed)
  static Eigen::MatrixXd getParametersSpace(const Eigen::MatrixXd & input_limits,
                                            const Eigen::MatrixXd & output_limits,
                                            bool narrow_slope);

  /// Return default parameters which can be used as an initial guess for a
  /// black-box optimizer given the provided limits 
  static Eigen::VectorXd getDefaultParameters(const Eigen::MatrixXd & input_limits,
                                              const Eigen::MatrixXd & output_limits);

protected:

  /// bias: B in Y = A*X + B
  Eigen::VectorXd bias;

  /// coefficients of the squared matrix A in Y=A*X + B
  Eigen::MatrixXd coeffs;
};

}
