#!/usr/bin/env python3
"""Build a valid B601+LinkerHand O6 URDF from the current RS SDK model.

The arm source is authoritative for links 1..6. The adapter and O6 subtree are
copied from the existing valid mechanical assembly; the old RS two-finger
subtree is removed. Fixed palm/TCP frames are inserted inside <robot>.
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def children(root: ET.Element) -> dict[str, list[tuple[ET.Element, str]]]:
    result: dict[str, list[tuple[ET.Element, str]]] = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            result.setdefault(parent.get("link"), []).append((joint, child.get("link")))
    return result


def subtree(root: ET.Element, first: str) -> set[str]:
    graph, found, stack = children(root), set(), [first]
    while stack:
        link = stack.pop()
        if link in found:
            continue
        found.add(link)
        stack.extend(child for _joint, child in graph.get(link, []))
    return found


def build(arm_path: Path, assembly_path: Path, output_path: Path) -> None:
    arm_root = ET.parse(arm_path).getroot()
    assembly_root = ET.parse(assembly_path).getroot()

    # Remove the complete stock gripper subtree from the current SDK arm.
    remove_links = subtree(arm_root, "gripper_end")
    for link in list(arm_root.findall("link")):
        if link.get("name") in remove_links:
            arm_root.remove(link)
    for joint in list(arm_root.findall("joint")):
        parent, child = joint.find("parent"), joint.find("child")
        if child is not None and child.get("link") in remove_links:
            arm_root.remove(joint)
        elif parent is not None and parent.get("link") in remove_links:
            arm_root.remove(joint)

    # Import adapter -> hand descendants from the valid mechanical assembly.
    import_links = subtree(assembly_root, "adapter")
    source_links = {link.get("name"): link for link in assembly_root.findall("link")}
    for name in sorted(import_links):
        arm_root.append(copy.deepcopy(source_links[name]))
    for joint in assembly_root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        p = parent.get("link") if parent is not None else ""
        c = child.get("link") if child is not None else ""
        if c in import_links and (p == "link6" or p in import_links):
            arm_root.append(copy.deepcopy(joint))

    existing_materials = {m.get("name") for m in arm_root.findall("material")}
    for material in assembly_root.findall("material"):
        if material.get("name") not in existing_materials:
            arm_root.insert(0, copy.deepcopy(material))
            existing_materials.add(material.get("name"))

    def fixed_frame(link_name: str, joint_name: str, xyz: str, parent: str,
                    rpy: str = "0 0 0") -> None:
        ET.SubElement(arm_root, "link", {"name": link_name})
        joint = ET.SubElement(arm_root, "joint", {"name": joint_name, "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": link_name})
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})

    # Nominal CAD-derived frames. Physically calibrate before grasp execution.
    fixed_frame("o6_palm_center", "hand_base_to_o6_palm_center", "0.020 0 0.060", "hand_base_link")
    # Closure-centre position for fixed thumb-yaw raw=87. Its orientation is
    # independently derived from the actual thumb <-> index/middle closing
    # direction. TCP axes remain [custom Z, custom X, custom Y].
    fixed_frame(
        "o6_grasp_tcp", "hand_base_to_o6_grasp_tcp",
        "0.0600 0.0275 0.1187", "hand_base_link",
        rpy="1.46867565 -0.72833526 0.36021895",
    )
    fixed_frame("gripper_end", "o6_grasp_tcp_to_gripper_end", "0 0 0", "o6_grasp_tcp")

    ET.indent(arm_root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(arm_root).write(output_path, encoding="unicode", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.arm, args.assembly, args.output)


if __name__ == "__main__":
    main()
