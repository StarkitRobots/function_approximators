#include "rhoban_fa/fa_tree.h"

#include "rhoban_fa/function_approximator_factory.h"
#include "rhoban_fa/split_factory.h"

namespace rhoban_fa
{

FATree::FATree()
{
}

FATree::FATree(std::unique_ptr<Split> split_,
               std::vector<std::unique_ptr<FunctionApproximator>> & children_)
{
  split = std::move(split_);
  children.resize(children_.size());
  for (size_t child_id = 0; child_id < children_.size(); child_id++)
  {
    children[child_id] = std::move(children_[child_id]);
  }
  children_.clear();
}


FATree::~FATree()
{
}

const Split & FATree::getSplit() const {
  return *split;
}

std::unique_ptr<FunctionApproximator> FATree::clone() const {
  int nb_children = split->getNbElements();
  std::unique_ptr<Split> split_copy = split->clone();
  std::vector<std::unique_ptr<FunctionApproximator>> children_copy(nb_children);
  for (int i = 0; i < nb_children; i++) {
    children_copy[i] = children[i]->clone();
  }
  FATree * tree = new FATree(std::move(split_copy),children_copy);
  tree->updateNodesCount();
  return std::unique_ptr<FunctionApproximator>(tree);
}

int FATree::getOutputDim() const
{
  checkConsistency("FATree::getOutputDim");
  return children[0]->getOutputDim();
}

void FATree::addSpaces(const Eigen::MatrixXd & global_space,
                       std::vector<Eigen::MatrixXd> * spaces) const{
  std::vector<Eigen::MatrixXd> sub_spaces = split->splitSpace(global_space);
  for (unsigned int idx = 0; idx < sub_spaces.size(); idx++) {
    // If 'child' is a tree, then dig deeper
    FATree * child_node = dynamic_cast<FATree*>(children[idx].get());
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
  FATree * child_node = dynamic_cast<FATree*>(children[index].get());
  if (child_node) {
    return child_node->getLeafApproximator(point);
  }
  else {
    return *(children[index]);
  }
}

const FATree &
FATree::getPreLeafApproximator(const Eigen::VectorXd & point) const {
  int index = split->getIndex(point);
  // If 'child' is a tree, then dig deeper
  FATree * child_node = dynamic_cast<FATree*>(children[index].get());
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
  FATree * child_node = dynamic_cast<FATree*>(children[index].get());
  if (child_node) {
    child_node->replaceApproximator(point,std::move(fa));
  }
  else {
    children[index] = std::move(fa);
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
  children[split->getIndex(input)]->predict(input, mean, covar);
}

void FATree::gradient(const Eigen::VectorXd & input,
                      Eigen::VectorXd & gradient) const
{
  checkConsistency("FATree::predict");
  children[split->getIndex(input)]->gradient(input, gradient);
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

int FATree::getNodesCount() const
{
  int total = 1;
  for (int count : children_sizes) {
    total += count;
  }
  return total;
}

void FATree::updateNodesCount()
{
  children_sizes.clear();
  for (size_t child_id = 0; child_id < children.size(); child_id++) {
    FATree * child_node = dynamic_cast<FATree*>(children[child_id].get());
    if (child_node != nullptr) {
      child_node->updateNodesCount();
      children_sizes.push_back(child_node->getNodesCount());
    } else {
      children_sizes.push_back(1);
    }
  }
}

std::vector<int> FATree::getLeavesId() const
{
  std::vector<int> result;
  fillLeavesId(&result, 0);
  return result;
}

int FATree::getLeafId(const Eigen::VectorXd & point) const
{
  int child_idx = split->getIndex(point);
  int index = 1;// Counting current node
  for (int i = 0; i < child_idx; i++) {
    index += children_sizes[i];
  }
  FATree * child_node = dynamic_cast<FATree*>(children[child_idx].get());
  if (child_node) {
    index += child_node->getLeafId(point);
  }
  return index;
}


void FATree::fillLeavesId(std::vector<int> * leaves_id, int offset) const
{
  offset++;// Current node is counted
  for (size_t child_id = 0; child_id < children.size(); child_id++) {
    FATree * child_node = dynamic_cast<FATree*>(children[child_id].get());
    if (child_node != nullptr) {
      child_node->fillLeavesId(leaves_id, offset);
    } else {
      leaves_id->push_back(offset);
    }
    offset += children_sizes[child_id];
  }
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
  for (size_t i = 0; i < children.size(); i++)
  {
    bytes_written += children[i]->write(out);
  }
  return bytes_written;
}

int FATree::read(std::istream & in)
{
  // Read split and then read children
  int bytes_read = 0;
  bytes_read += SplitFactory().read(in, split);
  children.resize(split->getNbElements());
  for (size_t i = 0; i < children.size(); i++)
  {
    bytes_read += FunctionApproximatorFactory().read(in, children[i]);
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
    oss << children[i]->toString();
    if (i < n -1) oss << "|";
  }
  oss << "))";
  return oss.str();
}

void FATree::checkConsistency(const std::string & caller_name) const
{
  if (!split)
    throw std::logic_error(caller_name + ": split has not been set");
  if (children.size() == 0)
    throw std::logic_error(caller_name + ": no children found");
  if ((int)children.size() != split->getNbElements())
  {
    std::ostringstream oss;
    oss << caller_name << ": number of children do no match split expected elements ("
        << children.size() << " children found, expecting " << split->getNbElements() << ")";
    throw std::logic_error(oss.str());
  }
}

}
