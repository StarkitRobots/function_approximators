#include "rosban_fa/fake_split.h"

#include "rosban_utils/io_tools.h"

namespace rosban_fa
{

FakeSplit::FakeSplit() : {}

int FakeSplit::getNbElements() const {
  return 1;
}

int FakeSplit::getIndex(const Eigen::VectorXd & input) const {
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
}

int FakeSplit::read(std::istream & in) {
  // Nothing needs to be read
  (void) in;
  return 0;
}

}
