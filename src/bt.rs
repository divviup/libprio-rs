// SPDX-License-Identifier: MPL-2.0

//! Types for defining binary trees.

type SubTree<V> = Option<Box<Node<V>>>;

/// Represents a node of a binary tree.
pub(crate) struct Node<V> {
    pub(crate) value: V,
    pub(crate) left: SubTree<V>,
    pub(crate) right: SubTree<V>,
}

impl<V> Node<V> {
    pub(crate) fn new(value: V) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }
}

/// Represents an append-only binary tree.
pub(crate) struct BinaryTree<V> {
    pub(crate) root: SubTree<V>,
}

impl<V> Default for BinaryTree<V> {
    fn default() -> Self {
        Self {
            root: Option::default(),
        }
    }
}
