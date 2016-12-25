#include "rosban_fa/fa_tree.h"

#include "rosban_fa/function_approximator_factory.h"
#include "rosban_fa/split_factory.h"

namespace rosban_fa
{

FATree::FATree()
{
}

FATree::FATree(std::unique_ptr<Split> split_,
               std::vector<std::unique_ptr<FunctionApproximator>> & childs_)
{
  split = std::move(split_);
  childs.resize(childs_.size());
  for (size_t child_id = 0; child_id < childs_.size(); child_id++)
  {
    childs[child_id] = std::move(childs_[child_id]);
  }
  childs_.clear();
}


FATree::~FATree()
{
}

std::unique_ptr<FATree> FATree::clone() const {
  //TODO: implement something more appropriate (real clone template for both:
  //      split and function_approximator)
  std::string tmp_path("/tmp/fa_tree_clone.data");
  this->save(tmp_path);
  std::unique_ptr<FunctionApproximator> ptr;
  FunctionApproximatorFactory().loadFromFile(tmp_path, ptr);
  return std::unique_ptr<FATree>(static_cast<FATree*>(ptr.release()));
}

int FATree::getOutputDim() const
{
  checkConsistency("FATree::getOutputDim");
  return childs[0]->getOutputDim();
}

void FATree::replaceApproximator(const Eigen::VectorXd & point,
                                 std::unique_ptr<FunctionApproximator> fa) {
  int index = split->getIndex(point);
  // If 'child' is a tree, then replace at next level
  FATree * child_node = dynamic_cast<FATree*>(childs[index].get());
  if (child_node) {
    child_node->replaceApproximator(point,std::move(fa));
  }
  else {
    childs[index] = std::move(fa);
  }
}

std::unique_ptr<FATree>
FATree::copyAndReplaceLeaf(const Eigen::VectorXd & point,
                           std::unique_ptr<FunctionApproximator> fa) const {
  std::unique_ptr<FATree> copy = clone();
  copy->replaceApproximator(point,std::move(fa));
  return std::move(copy);
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
  // Read split and then read childs
  int bytes_read = 0;
  bytes_read += SplitFactory().read(in, split);
  childs.resize(split->getNbElements());
  for (size_t i = 0; i < childs.size(); i++)
  {
    bytes_read += FunctionApproximatorFactory().read(in, childs[i]);
  }
  // Check after reading that consistency is ensured
  checkConsistency("FATree::read");
  return bytes_read;
}

void FATree::checkConsistency(const std::string & caller_name) const
{
  if (!split)
    throw std::logic_error(caller_name + ": split has not been set");
  if (childs.size() == 0)
    throw std::logic_error(caller_name + ": no childs found");
  if ((int)childs.size() != split->getNbElements())
  {
    std::ostringstream oss;
    oss << caller_name << ": number of childs do no match split expected elements ("
        << childs.size() << " childs found, expecting " << split->getNbElements() << ")";
    throw std::logic_error(oss.str());
  }
}

}
