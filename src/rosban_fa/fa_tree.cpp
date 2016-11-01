#include "rosban_fa/fa_tree.h"

#include "rosban_fa/function_approximator_factory.h"
#include "rosban_fa/split_factory.h"

namespace rosban_fa
{

FATree::FATree()
{
}

FATree::~FATree()
{
}

int FATree::getOutputDim() const
{
  checkConsistency("FATree::getOutputDim");
  return childs[0]->getOutputDim();
}

void FATree::predict(const Eigen::VectorXd & input,
                     Eigen::VectorXd & mean,
                     Eigen::MatrixXd & covar) const
{
  checkConsistency("FATree::predict");
  childs[split->getIndex(input)]->predict(input, mean, covar);
}

void FATree::gradient(const Eigen::VectorXd & input,
                      Eigen::VectorXd & gradient) const
{
  checkConsistency("FATree::predict");
  childs[split->getIndex(input)]->gradient(input, gradient);
}

void FATree::getMaximum(const Eigen::MatrixXd & limits,
                        Eigen::VectorXd & input,
                        double & output) const
{
  (void)limits;
  (void)input;
  (void)output;
  throw std::logic_error("FATree::getMaximum: unimplemented method");
}

int FATree::getClassID() const
{
  return ID::FATree;
}

int FATree::writeInternal(std::ostream & out) const
{
  checkConsistency("FATree::writeInternal");
  int bytes_written = 0;
  bytes_written += split->write(out);
  for (size_t i = 0; i < childs.size(); i++)
  {
    bytes_written += childs[i]->write(out);
  }
  return bytes_written;
}

int FATree::read(std::istream & in)
{
  // Check after reading that consistency is ensured
  checkConsistency("FATree::read");
  // Read split and then read childs
  int bytes_read = 0;
  bytes_read += SplitFactory().read(in, split);
  childs.resize(split->getNbElements());
  for (size_t i = 0; i < childs.size(); i++)
  {
    bytes_read += FunctionApproximatorFactory().read(in, childs[i]);
  }
  return bytes_read;
}

void FATree::checkConsistency(const std::string & caller_name) const
{
  if (!split)
    throw std::logic_error(caller_name + ": split has not been set");
  if (childs.size() == 0)
    throw std::logic_error(caller_name + ": no childs found");
  if (childs.size() != split->getNbElements())
  {
    std::ostringstream oss;
    oss << caller_name << ": number of childs do no match split expected elements ("
        << childs.size() << " childs found, expecting " << split->getNbElements() << ")";
    throw std::logic_error(oss.str());
  }
}

}
