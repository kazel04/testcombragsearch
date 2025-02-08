import ast
import os
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState


class Node:
    def __init__(self, name: str, node_type: str, level: int):
        self.name = name
        self.node_type = node_type
        self.level = level
        self.children: list[Node] = []

    def add_child(self, node: 'Node'):
        self.children.append(node)


class ProjectAnalyzer(ast.NodeVisitor):
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.root_node = Node(project_root, "project", 0)
        self.current_module: Node | None = None
        self.current_class: Node | None = None
        self.current_file: str | None = None
        self.node_count: str = 1

    def visit_Module(self, node: ast.Module) -> None:
        module_name = os.path.basename(self.current_file)
        module_node = Node(module_name, "module", 1)
        self.root_node.add_child(module_node)
        self.current_module = module_node
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_node = Node(node.name, "class", 2)
        self.current_module.add_child(class_node)
        self.current_class = class_node
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        function_node = Node(node.name, "function", 3)
        if self.current_class:
            self.current_class.add_child(function_node)
        else:
            self.current_module.add_child(function_node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def analyze(self) -> None:
        for root, _, files in os.walk(self.project_root):
            
            for file in files:
                if file.endswith(".py") and not file == "activationscript.py":
                    self.node_count+=1
                    nodes.append(StreamlitFlowNode(str(self.node_count), 
                            (250, 200*(self.node_count-1)/len(files)), 
                            {'content': file}, 
                            'default', 
                            'right', 
                            'left', 
                            draggable=True))
                    edges.append(StreamlitFlowEdge(f'1-{str(self.node_count)}', '1', str(self.node_count), animated=True, marker_end={'type': 'arrow'}))
                    
                    file_path = os.path.join(root, file)
                    self.current_file = file_path
                    with open(file_path, "r") as f:
                        code = f.read()
                        tree = ast.parse(code)
                        self.visit(tree)
    
    def print_tree(self, node: Node, indent: str = "") -> None:
        self.node_count+=1
        
        print(f"{indent}{node.name}")
                  
        for child in node.children:
            print(node.children)
            self.print_tree(child, indent + "    ")
"""
nodes.append(StreamlitFlowNode(str(self.node_count), 
                    (250, 200*(self.node_count-1)/len(files)), 
                    {'content': node.name}, 
                    'default', 
                    'right', 
                    'left', 
                    draggable=False))
edges.append(StreamlitFlowEdge(f'1-{str(self.node_count)}', '1', str(self.node_count), animated=True, marker_end={'type': 'arrow'}))
"""  

def visualize_project(project_root: str) -> None:
    analyzer = ProjectAnalyzer(project_root)
    analyzer.analyze()
    analyzer.print_tree(analyzer.root_node)

if __name__ == "__main__":
    project_root = "sepmanalyzer"
    nodes = [StreamlitFlowNode( id='1', 
                            pos=(100, 100), 
                            data={'content': project_root}, 
                            node_type='input', 
                            source_position='right', 
                            draggable=False)]
    edges=[]
    visualize_project(project_root)
    #nodes.extend(analyzed_nodes)
    #edges.extend(analyzed_edges)
    print(nodes, edges)

    state = StreamlitFlowState(nodes, edges)

    streamlit_flow('static_flow',
                    state,
                    fit_view=True,
                    show_minimap=False,
                    show_controls=False,
                    pan_on_drag=False,
                    allow_zoom=False)
    