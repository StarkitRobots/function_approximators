#pragma once

#include "rhoban_fa/split.h"

namespace rhoban_fa
{
/// A linear split is a split along an hyperplan: one side has index 0, the other has index 1
///
/// The hyperplane is defined by sum(coeffs_i*x_i) + offset = 0 with a_i the i-th
/// row of coeffs, b the offset and x_i
class LinearSplit : public Split
{
public:
  LinearSplit();

  /// If coeffs is not normalized, then normalizes both coeffs and offset in the process
  LinearSplit(const Eigen::VectorXd& coeffs, double offset);

  /// Return the split based on hyperplane with coefficients 'coeffs', passing by the
  /// specified point.
  /// coeffs do not need to be normalized
  LinearSplit(const Eigen::VectorXd& coeffs, const Eigen::VectorXd& point);

  virtual std::unique_ptr<Split> clone() const override;

  virtual int getNbElements() const override;

  virtual int getIndex(const Eigen::VectorXd& input) const override;

  virtual std::vector<Eigen::MatrixXd> splitSpace(const Eigen::MatrixXd& space) const;

  virtual std::string toString() const override;

  // Stream Serialization
  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream& out) const override;
  virtual int read(std::istream& in) override;

private:
  /// Coefficients of the director vector
  Eigen::VectorXd coeffs;

  /// Distance to the origin point
  double offset;
};

}  // namespace rhoban_fa
