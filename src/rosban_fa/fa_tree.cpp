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

const Split & FATree::getSplit() const {
  return *split;
}

std::unique_ptr<FunctionApproximator> FATree::clone() const {
  int nb_childs = split->getNbElements();
  std::unique_ptr<Split> split_copy = split->clone();
  std::vector<std::unique_ptr<FunctionApproximator>> childs_copy(nb_childs);
  for (int i = 0; i < nb_childs; i++) {
    childs_copy[i] = childs[i]->clone();
  }
  std::unique_ptr<FunctionApproximator> copy(new FATree(std::move(split_copy),
                                                        childs_copy));
  return std::move(copy);
}

int FATree::getOutputDim() const
{
  checkConsistency("FATree::getOutputDim");
  return childs[0]->getOutputDim();
}

void FATree::addSpaces(const Eigen::MatrixXd & global_space,
                       std::vector<Eigen::MatrixXd> * spaces) const{
  std::vector<Eigen::MatrixXd> sub_spaces = split->splitSpace(global_space);
  for (unsigned int idx = 0; idx < sub_spaces.size(); idx++) {
    // If 'child' is a tree, then dig deeper
    FATree * child_node = dynamic_cast<FATree*>(childs[idx].get());
    if (child_node) {
      child_node->addSpaces(sub_spaces[idx], spaces);
    }
    else {
      spaces->push_back(sub_spaces[idx]);
    }
  }
}

const FunctionApproximator &
FATree::getLeafApproximator(const Eigen::VectorXd & point) const {
  int index = split->getIndex(point);
  // If 'child' is a tree, then dig deeper
  FATree * child_node = dynamic_cast<FATree*>(childs[index].get());
  if (child_node) {
    return child_node->getLeafApproximator(point);
  }
  else {
    return *(childs[index]);
  }
}

const FATree &
FATree::getPreLeafApproximator(const Eigen::VectorXd & point) const {
  int index = split->getIndex(point);
  // If 'child' is a tree, then dig deeper
  FATree * child_node = dynamic_cast<FATree*>(childs[index].get());
  if (child_node) {
    return child_node->getPreLeafApproximator(point);
  }
  else {
    return *this;
  }
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
  // clone and cast clone to FATree
  std::unique_ptr<FATree> copy(static_cast<FATree *>(clone().release()));
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


std::string FATree::toString() const {
  std::ostringstream oss;
  oss << "(FATree|Split: " << split->toString()
      << "|Approximators:(";
  int n = split->getNbElements();
  for (int i = 0; i < n;i++) {
    oss << childs[i]->toString();
    if (i < n -1) oss << "|";
  }
  oss << "))";
  return oss.str();
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
