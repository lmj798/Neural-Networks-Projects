from typing import Dict, List
from tensor import Tensor, Value

def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        grad_list = node_to_output_grads_list[node]
        if len(grad_list) == 1:
            node.grad = grad_list[0]
        else:
            grad_sum = grad_list[0]
            for g in grad_list[1:]:
                grad_sum = grad_sum + g
            node.grad = grad_sum

        if node.is_leaf():
            continue

        input_grads = node.op.gradient_as_tuple(node.grad, node)
        for i, grad in enumerate(input_grads):
            input_node = node.inputs[i]
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            node_to_output_grads_list[input_node].append(grad)

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    visited = set()
    topo_order = []
    topo_sort_dfs(node_list[-1], visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for pre_node in node.inputs:
        topo_sort_dfs(pre_node, visited, topo_order)
    topo_order.append(node)