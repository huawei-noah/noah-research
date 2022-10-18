# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OP_LIST = ["+", "-", "*", "/", "^"]
ORDER_DICT = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
import random

# An expression tree node
class Et:
    # Constructor to create a node
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.father = None

    def __str__(self):
        postfix = decode_postfix_from_tree(self)
        infix = from_postfix_to_infix(postfix)
        return infix

    def tree_update(self, value):
        """ update self with new value """
        if self.value in OP_LIST:
            assert value in OP_LIST, "should update an operator with another operator"
        else:
            assert value not in OP_LIST, "should update a number with another number"
        self.value = value
        return self.find_root()

    def tree_exchange(self):
        """ exchange the children """
        assert self.value in OP_LIST, "only children of operators can be exchanged"
        self.left, self.right = self.right, self.left
        return self.find_root()

    def tree_delete(self, delete_child):
        """ delete this self and one of its children """
        assert self.value in OP_LIST, "only operators can be deleted"
        assert delete_child in ["left", "right"]
        remain = self.right if delete_child == "left" else self.left
        father = self.father
        self.father = None

        if father:
            if father.left is self:
                father.left = remain
            else:
                assert father.right is self
                father.right = remain
        remain.father = father
        return remain.find_root()

    def tree_add(self, add_op, add_value, add_value_at):
        """ add a new operator with new value on this node """
        assert add_op in OP_LIST, "add_op should be an operator"
        new_father = Et(add_op)
        new_child = Et(add_value)
        new_child.father = new_father
        assert add_value_at in ["left", "right"]
        if add_value_at == "left":
            new_father.left = new_child
            new_father.right = self
        else:
            new_father.left = self
            new_father.right = new_child
        prev_father = self.father
        new_father.father = prev_father
        self.father = new_father
        if prev_father:
            if prev_father.left is self:
                prev_father.left = new_father
            else:
                assert prev_father.right is self
                prev_father.right = new_father
        return new_father.find_root()

    def find_root(self):
        node = self
        while node.father:
            node = node.father
        return node

    def get_sub_nodes(self):
        if self.value is None:
            return []
        left, right = [], []
        if self.left:
            left = self.left.get_sub_nodes()
        if self.right:
            right = self.right.get_sub_nodes()
        return left + [self] + right

def decode_postfix_from_tree(root):
    if root is None:
        return ""
    tmp = ""
    if root.left:
        tmp = decode_postfix_from_tree(root.left)
    if root.right:
        tmp = tmp + " " + decode_postfix_from_tree(root.right) if tmp else decode_postfix_from_tree(root.right)
    tmp = tmp + " " + root.value if tmp else root.value
    return tmp

# Returns root of constructed tree for given postfix expression
def construct_exp_tree_from_postfix(postfix):
    if isinstance(postfix, str):
        postfix = postfix.split(' ')
    stack = []

    # Traverse through every character of input expression
    for char in postfix:

        # if operand, simply push into stack
        if char not in ["+", "-", "*", "/", "^"]:
            t = Et(char)
            stack.append(t)
        # Operator
        else:
            # Pop two top nodes
            t = Et(char)
            t1 = stack.pop()
            t2 = stack.pop()

            # make them children
            t.right = t1
            t.left = t2

            # give their father
            t1.father = t
            t2.father = t

            # Add this subexpression to stack
            stack.append(t)
    # Only element  will be the root of expression tree
    t = stack.pop()

    return t


def from_infix_to_postfix(expression):
    if isinstance(expression, str):
        expression = expression.split(' ')
        expression = [e for e in expression if e]
    st = list()
    res = list()
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in ORDER_DICT:
            while len(st) > 0 and st[-1] not in ["(", "["] and ORDER_DICT[e] <= ORDER_DICT[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return " ".join(res)


def from_postfix_to_infix(postfix):
    if isinstance(postfix, str):
        postfix = postfix.split(' ')
    stack = []
    for elem in postfix:
        if elem in OP_LIST:
            a, od_a = stack.pop()
            b, od_b = stack.pop()
            od_c = ORDER_DICT[elem]
            if od_a <= od_c:
                a = "( " + a + " )"
            if od_b < od_c:
                b = "( " + b + " )"
            tmp = b + " " + elem + " " + a
            stack.append((tmp, od_c))
        else:
            stack.append((elem, 3))
    assert len(stack) == 1
    return stack[-1][0]


def corrupt_add(node, num_list):
    add_op = OP_LIST[random.randint(0, 3)]
    add_value = random.choice(num_list)
    add_value_at = "left" if random.random()<0.5 else "right"
    return node.tree_add(add_op, add_value, add_value_at)

def corrupt_update(node, num_list):
    prev_num = node.value
    assert prev_num in num_list
    candidates = list(set(num_list) - {prev_num})
    if len(candidates) > 0:
        return node.tree_update(random.choice(candidates))
    return node.find_root()

def corrupt_exchange(node):
    return node.tree_exchange()

def corrupt_delete(node):
    father = node.father
    if father.left is node:
        return father.tree_delete("left")
    else:
        return father.tree_delete("right")

def one_step_corrupt(root, num_list=None):
    if root.value is None:
        return root

    node_list = root.get_sub_nodes()
    if num_list is None:
        num_list = []
        for node in node_list:
            if node.value not in OP_LIST:
                num_list.append(node.value)

    if root.value not in OP_LIST: #only one node, so apply tree_add or update
        root = corrupt_add(root, num_list)
    else:
        corrupt_type = random.choice(["add", "delete", "update", "exchange"])
        if corrupt_type == "add":
            node = random.choice(node_list)
            root = corrupt_add(node, num_list)
        elif corrupt_type == "update":
            node = random.choice(node_list)
            if node.value in OP_LIST:
                if node.value == "^":
                    root = corrupt_update(node, OP_LIST)
                else:
                    root = corrupt_update(node, OP_LIST[:-1])
            else:
                root = corrupt_update(node, num_list)
        elif corrupt_type == "delete":
            # only nums can be deleted or updated in one step
            node = random.choice(node_list)
            while node.value in OP_LIST:
                node = random.choice(node_list)
            root = corrupt_delete(node)
        else:
            assert corrupt_type == "exchange"
            #only ops can be exchanged
            node = random.choice(node_list)
            while node.value not in OP_LIST:
                node = random.choice(node_list)
            root = corrupt_exchange(node)
    return root

def corrupt_expression(expression):
    post = from_infix_to_postfix(expression)
    tree = construct_exp_tree_from_postfix(post)
    node_list = tree.get_sub_nodes()

    num_list = []
    for node in node_list:
        if node.value not in OP_LIST:
            num_list.append(node.value)

    tree = one_step_corrupt(tree, num_list)
    corrupt_str = tree.__str__()
    return corrupt_str

if __name__ == "__main__":
    for i in range(10):
        exp = "#1 + 3 + 4"
        print(corrupt_expression(exp))



