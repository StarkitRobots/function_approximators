#pragma once

#include "rosban_fa/split.h"

namespace rosban_fa
{

/// Fake split have only one child, they are used in order to create a fake root
/// in some algorithms
class FakeSplit : public Split
{
public:

  FakeSplit();

  virtual int getNbElements() const override;

  virtual int getIndex(const Eigen::VectorXd & input) const override;

  virtual std::vector<Eigen::MatrixXd> splitSpace(const Eigen::MatrixXd & space) const;

  // Stream Serialization
  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;

  virtual std::string toString() const override;
};

}
