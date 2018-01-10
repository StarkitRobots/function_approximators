#include "rosban_fa/fake_split.h"

#include "rhoban_utils/io_tools.h"

namespace rosban_fa
{

FakeSplit::FakeSplit() {}

std::unique_ptr<Split> FakeSplit::clone() const {
  return std::unique_ptr<Split>(new FakeSplit());
}

int FakeSplit::getNbElements() const {
  return 1;
}

int FakeSplit::getIndex(const Eigen::VectorXd & input) const {
  (void) input;
  return 0;
}

std::vector<Eigen::MatrixXd>
FakeSplit::splitSpace(const Eigen::MatrixXd & space) const {
  return {space};
}

std::string FakeSplit::toString() const {
  return "(FakeSplit)";
}


int FakeSplit::getClassID() const {
  return ID::Fake;
}

int FakeSplit::writeInternal(std::ostream & out) const {
  // Nothing needs to be written
  (void) out;
  return 0;
}

int FakeSplit::read(std::istream & in) {
  // Nothing needs to be read
  (void) in;
  return 0;
}

}
