#pragma once

#include "starkit_fa/split.h"

namespace starkit_fa
{
/// A point split is a split along all dimensions, following this idea:
/// index = Sum((input(dim) > point(dim)) * 2^dim):
class PointSplit : public Split
{
public:
  PointSplit();
  PointSplit(const Eigen::VectorXd& point);

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
  Eigen::VectorXd split_point;
};

}  // namespace starkit_fa
