import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
from typing import Dict, List, Any, Union

__all__ = ["xml_to_dict", "dict_to_xml", "write_xml", "pretty_xml"]


def xml_to_dict(xml_file: str) -> Dict[str, Any]:
    """
    Recursively parses xml contents to python dict.

    NOTE: `object` is the only tag that can appear multiple times.
    """
    assert os.path.exists(xml_file), f"File does not exist: {xml_file}"
    return _xml_to_dict_helper(ET.parse(xml_file).getroot())


def _xml_to_dict_helper(node: ET.Element) -> Dict[str, Any]:
    """ Helper function to parse xml_to_dict."""
    if len(node) == 0:
        return {node.tag: node.text}
    node_dict = {}
    for child in node:
        child_dict = _xml_to_dict_helper(child)
        if child.tag != "object":
            node_dict[child.tag] = child_dict[child.tag]
        else:
            node_dict.setdefault(child.tag, []).append(child_dict[child.tag])
    return {node.tag: node_dict}


def dict_to_xml(xml_dict: Dict[str, Any]) -> ET.Element:
    """Recursively parses a xml_dict to xml element tree."""
    assert isinstance(xml_dict, dict), f"Wrong type: type(xml_dict)={type(xml_dict)}"
    if "annotation" in xml_dict:
        xml_dict = xml_dict["annotation"]
    return _dict_to_xml_helper("annotation", xml_dict)


def _dict_to_xml_helper(tag: str, node: Union[str, Dict, List]) -> Union[ET.Element, List[ET.Element]]:
    """Helper function to parse xml_dict."""
    if isinstance(node, (str, int, float, type(None))):  # Elements with primal type are leaf nodes
        child = ET.Element(tag)
        child.text = str(node)
        return child
    if isinstance(node, dict):
        child = ET.Element(tag)
        for node_tag, sub_node in node.items():
            if node_tag != "object":
                child.append(_dict_to_xml_helper(node_tag, sub_node))
            else:
                for sub_child in _dict_to_xml_helper(node_tag, sub_node):
                    child.append(sub_child)
        return child
    if isinstance(node, list):
        children = []
        for sub_node in node:
            sub_child = _dict_to_xml_helper(tag, sub_node)
            children.append(sub_child)
        return children
    raise ValueError(f"[Error] Undefined node:{node} type(node)=={type(node)}")


def write_xml(file: str, node: Union[ET.ElementTree, ET.Element]):
    """Write xml_tree to file."""
    if isinstance(node, ET.Element):
        xml_str = pretty_xml(node)
        with open(file, 'w') as f:
            f.write(xml_str)
    elif isinstance(node, ET.ElementTree):
        with open(file, 'wb') as f:
            node.write(f)
    else:
        raise ValueError(f"Unknown argument `node`: type(node)==f{type(node)}!")


def pretty_xml(node: ET.Element) -> str:
    """Return a pretty indented xml string given a ET.Element node."""
    assert isinstance(node, ET.Element), f"Wrong type: type(node)=={type(node)}"
    raw_xml_str = ET.tostring(node)
    pretty_str = MD.parseString(raw_xml_str).toprettyxml()
    return pretty_str


if __name__ == '__main__':
    from pprint import pprint

    # XML to DICT
    xml_file = r"/home/sparkai/PycharmProjects/GNetDet_MDK_Pytorch_2021_09_02/data/Smoking/VOC2007/Annotations/1.xml"
    xml_dict = xml_to_dict(xml_file)
    pprint(xml_dict)
    print()

    # DICT TO XML
    xml_root = dict_to_xml(xml_dict)
    print(xml_root)
    print()

    # Print Pretty XML
    xml_str = pretty_xml(xml_root)
    print(xml_str)
    print()
