#!/usr/bin/env python3
"""Remove large machine-generated branches from a RenLib .lib file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys


HEADER_SIZE = 20
MAGIC = b"\xffRenLib\xff"
LEAF_FLAG = 0x40
SIBLING_FLAG = 0x80
TEXT_FLAG = 0x01
COMMENT_FLAG = 0x08


class LibFormatError(ValueError):
    """The input does not contain a complete RenLib tree."""


@dataclass
class Node:
    move: int
    flag: int
    payload: bytes
    children: list["Node"] = field(default_factory=list)


def read_entry(data: bytes, offset: int) -> tuple[Node, int]:
    """Read one entry and preserve every byte other than tree flags."""
    if offset + 2 > len(data):
        raise LibFormatError(f"entry at offset {offset} is truncated")

    move = data[offset]
    flag = data[offset + 1]
    payload_start = offset + 2
    offset = payload_start

    if flag & TEXT_FLAG:
        if offset + 2 > len(data):
            raise LibFormatError(f"text extension at offset {offset} is truncated")
        offset += 2

    for has_string in (flag & COMMENT_FLAG, flag & TEXT_FLAG):
        if not has_string:
            continue
        string_end = data.find(b"\0", offset)
        if string_end < 0:
            raise LibFormatError(f"string at offset {offset} is not NUL-terminated")
        raw_length = string_end - offset
        offset = string_end + 1
        if (raw_length + 1) % 2:
            if offset >= len(data):
                raise LibFormatError(f"string padding at offset {offset} is truncated")
            offset += 1

    return Node(move, flag, data[payload_start:offset]), offset


def read_node(data: bytes, offset: int) -> tuple[Node, int]:
    node, offset = read_entry(data, offset)
    if not node.flag & LEAF_FLAG:
        child, offset = read_node(data, offset)
        node.children.append(child)
        while child.flag & SIBLING_FLAG:
            child, offset = read_node(data, offset)
            node.children.append(child)
    return node, offset


def read_tree(data: bytes) -> tuple[bytes, list[Node]]:
    if len(data) < HEADER_SIZE or data[: len(MAGIC)] != MAGIC:
        raise LibFormatError("not a RenLib .lib file")

    header = data[:HEADER_SIZE]
    offset = HEADER_SIZE
    roots: list[Node] = []
    while offset < len(data):
        node, offset = read_node(data, offset)
        roots.append(node)
        if not node.flag & SIBLING_FLAG:
            break
    if offset != len(data):
        raise LibFormatError(f"unexpected trailing data at offset {offset}")
    return header, roots


def count_descendants(node: Node) -> int:
    return sum(1 + count_descendants(child) for child in node.children)


def prune_large_branches(
    nodes: list[Node], threshold: int, move_count: int = 0
) -> tuple[int, int]:
    """Prune large branches only from positions at move 10 or later."""
    cut_positions = 0
    removed_nodes = 0
    for node in nodes:
        next_move_count = move_count + (node.move != 0)
        if next_move_count >= 10 and len(node.children) >= threshold:
            cut_positions += 1
            removed_nodes += sum(count_descendants(child) + 1 for child in node.children)
            node.children.clear()
        else:
            cuts, removed = prune_large_branches(node.children, threshold, next_move_count)
            cut_positions += cuts
            removed_nodes += removed
    return cut_positions, removed_nodes


def write_nodes(nodes: list[Node], output: bytearray) -> None:
    for index, node in enumerate(nodes):
        flag = node.flag & ~(LEAF_FLAG | SIBLING_FLAG)
        if not node.children:
            flag |= LEAF_FLAG
        if index + 1 < len(nodes):
            flag |= SIBLING_FLAG
        output.extend((node.move, flag))
        output.extend(node.payload)
        write_nodes(node.children, output)


def slim_file(input_path: Path, output_path: Path, threshold: int) -> tuple[int, int, int, int]:
    header, roots = read_tree(input_path.read_bytes())
    before_count = sum(count_descendants(root) + 1 for root in roots)
    cut_positions, removed_nodes = prune_large_branches(roots, threshold)
    output = bytearray(header)
    write_nodes(roots, output)
    output_path.write_bytes(output)
    return before_count, before_count - removed_nodes, cut_positions, removed_nodes


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_slim{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove descendants of RenLib positions at move 10 or later with many direct next moves."
    )
    parser.add_argument("input", type=Path, help="source .lib file")
    parser.add_argument("-o", "--output", type=Path, help="output .lib file")
    parser.add_argument(
        "-n", "--threshold", type=int, default=10,
        help="number of direct children that marks a position as machine-generated (default: 10; positions before move 10 are kept)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = (args.output or default_output_path(input_path)).resolve()

    if args.threshold < 1:
        print("error: --threshold must be at least 1", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2
    if input_path == output_path:
        print("error: output path must differ from the input path", file=sys.stderr)
        return 2
    if output_path.exists() and not args.force:
        print(f"error: output file already exists: {output_path} (use --force to overwrite)", file=sys.stderr)
        return 2

    try:
        before, after, positions, removed = slim_file(input_path, output_path, args.threshold)
    except (OSError, LibFormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"saved: {output_path}")
    print(f"positions: {before} -> {after} (removed {removed})")
    print(f"cut positions: {positions} (threshold: {args.threshold} direct children)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
