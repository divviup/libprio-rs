// SPDX-License-Identifier: MPL-2.0

//! An append-only binary tree.
//!
//! ## Properties:
//! - Binary tree: nodes have either 0, 1, or 2 child nodes.
//! - Append-only: tree only grows, values can only be inserted and queried.
//! - Serializable: tree can be coverted to and from bytes.
//!
//! ## Creation
//! Use [`BinaryTree::default`] to create a binary tree, note that there must be
//! one value at the root node.
//!
//! ## Insertion
//! Given a value and a binary path, [`BinaryTree::insert`] stores the value
//! at the end of the path.
//!
//! ## Query
//! Given a binary path, [`BinaryTree::get`] returns a reference to the value
//! stored in the node located at the end of the path. The empty path returns
//! the value at the root node. No value is returned if the end of the path
//! is unreachable.
//!
//! ## Serialization
//! A tree can be serializad to and from bytes using [`BinaryTree::encode`]
//! and [`BinaryTree::decode_with_param`] functions, respectively. One-byte
//! markers are used allowing to store sparse trees with a lower overhead.
//!
//! ## Example
//! This binary tree can be created with the following code:
//!
//! ```txt
//!               ()=@
//!          /              \
//!       (0)=a           (1)=b
//!     /      \         /      \
//! (00)=c  (01)=d   (10)=e  (11)=f
//! ```
//!
//! use prio::bt::BinaryTree;
//! use bitvec::{bits,prelude::Lsb0};
//! let mut tree = BinaryTree::default();
//! tree.insert(bits!(), '@');
//! tree.insert(bits!(0), 'a');
//! tree.insert(bits!(1), 'b');
//! tree.insert(bits!(0, 0), 'c');
//! tree.insert(bits!(0, 1), 'd');
//! tree.insert(bits!(1, 0), 'e');
//! tree.insert(bits!(1, 1), 'f');

// TODO(#947): Remove these lines once the module gets used by Mastic implementation.
#![allow(dead_code)]
use core::fmt::Debug;
use std::io::Cursor;

use bitvec::slice::BitSlice;

use crate::codec::{CodecError, Decode, Encode, ParameterizedDecode};

/// Errors triggered by binary tree operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BinaryTreeError<V> {
    /// Error when inserting in a node with existing value.
    #[error("node already contains a value")]
    InsertNonEmptyNode(V),
    /// Error when an operation cannot reach the node at the end of a path.
    #[error("unreachable node in the tree")]
    UnreachableNode(V),
}

/// Used to indicate a traversal path on the binary tree.
pub type Path = BitSlice;

type SubTree<V> = Option<Box<Node<V>>>;

/// Represents a node of a binary tree.
pub struct Node<V> {
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

    /// Inserts the value at the end of the path.
    ///
    /// This function traverses the tree from this node (`self`) until reaching the
    /// node at the end of the path. If the node is unreachable or already
    /// contains a value, an error wrapping the value is returned. Otherwise,
    /// a new node containing the value is inserted.
    ///
    /// # Returns
    /// - `Ok(())` when the node is inserted at the end of the path.
    /// - `Err(InsertNonEmptyNode(value))` when the subtree already contains a
    ///   value at the end of the path.
    /// - `Err(UnreachablePath(value))` when the end of the path is unreachable.
    ///
    /// In the error cases, no insertion occurs and the value is returned to
    /// the caller of this function.
    pub fn insert(&mut self, path: &Path, value: V) -> Result<(), BinaryTreeError<V>> {
        enum Ref<'a, V> {
            This(&'a mut Node<V>),
            Other(&'a mut Option<Box<Node<V>>>),
        }

        // Finds the node at the end of the path.
        let mut node = Ref::This(self);
        for bit in path.iter() {
            node = match node {
                Ref::This(n) => Ref::Other(if !bit { &mut n.left } else { &mut n.right }),
                Ref::Other(Some(n)) => Ref::Other(if !bit { &mut n.left } else { &mut n.right }),
                Ref::Other(None) => {
                    return Err(BinaryTreeError::UnreachableNode(value));
                }
            };
        }

        // Checks whether the node already has a value,
        // If so, returns an error wrapping the value that could not be inserted.
        // otherwise, a new node is created containing the value to be inserted.
        match node {
            Ref::This(_) | Ref::Other(Some(_)) => {
                return Err(BinaryTreeError::InsertNonEmptyNode(value))
            }
            Ref::Other(empty_node) => {
                *empty_node = Some(Box::new(Node::new(value)));
            }
        }

        Ok(())
    }

    /// Gets a reference to the value located at the end of the path.
    ///
    /// This function traverses the tree from this node (`self`) until reaching the
    /// node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent. Otherwise, it returns a reference to the
    /// value stored in the node.
    pub fn get(&self, path: &Path) -> Option<&V> {
        let mut node = self;
        for bit in path {
            match if !bit { &node.left } else { &node.right } {
                None => return None,
                Some(next_node) => node = next_node,
            };
        }
        Some(&node.value)
    }

    /// Gets a mutable reference to the node located at the end of the path.
    ///
    /// This function traverses the tree from this node (`self`) until reaching the
    /// node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent. Otherwise, it returns a mutable reference
    /// to the node.
    pub fn get_node(&mut self, path: &Path) -> Option<&mut Node<V>> {
        let mut node = self;
        for bit in path {
            match if !bit {
                &mut node.left
            } else {
                &mut node.right
            } {
                None => return None,
                Some(next_node) => node = next_node,
            };
        }
        Some(node)
    }
}

/// Represents an append-only binary tree.
pub struct BinaryTree<V> {
    pub(crate) root: SubTree<V>,
}

impl<V> BinaryTree<V> {
    /// Inserts the value at the end of the path.
    ///
    /// This function traverses the tree from the root node until reaching the
    /// node at the end of the path. If the node is unreachable or already
    /// contains a value, an error wrapping the value is returned. Otherwise,
    /// a new node containing the value is inserted.
    ///
    /// # Returns
    /// - `Ok(())` when the node is inserted at the end of the path.
    /// - `Err(InsertNonEmptyNode(value))` when the tree already contains a
    ///   value at the end of the path.
    /// - `Err(UnreachablePath(value))` when the end of the path is unreachable.
    ///
    /// In the error cases, no insertion occurs and the value is returned to
    /// the caller of this function.
    pub fn insert(&mut self, path: &Path, value: V) -> Result<(), BinaryTreeError<V>> {
        if let Some(node) = &mut self.root {
            node.insert(path, value)
        } else if path.is_empty() {
            self.root = Some(Box::new(Node::new(value)));
            Ok(())
        } else {
            Err(BinaryTreeError::UnreachableNode(value))
        }
    }

    /// Gets a reference to the value located at the end of the path.
    ///
    /// This function traverses the tree from the root node until reaching the
    /// node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent. Otherwise, it returns a reference to the
    /// value stored in the node.
    pub fn get(&self, path: &Path) -> Option<&V> {
        self.root.as_ref().and_then(|node| node.get(path))
    }

    /// Gets a mutable reference to the node located at the end of the path.
    ///
    /// This function traverses the tree from the root node until reaching the
    /// node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent. Otherwise, it returns a mutable reference
    /// to the node.
    pub fn get_node(&mut self, path: &Path) -> Option<&mut Node<V>> {
        self.root.as_mut().and_then(|node| node.get_node(path))
    }
}

impl<V> Default for BinaryTree<V> {
    fn default() -> Self {
        Self {
            root: Option::default(),
        }
    }
}

impl<V: Encode> Encode for Node<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.value.encode(bytes)?;

        let mut stack = Vec::new();
        stack.push(&self.right);
        stack.push(&self.left);

        // Nodes are stored following a pre-order traversal.
        while let Some(elem) = stack.pop() {
            match elem {
                None => CodecMarker::Leaf.encode(bytes)?,
                Some(node) => {
                    CodecMarker::Inner.encode(bytes)?;
                    node.value.encode(bytes)?;
                    stack.push(&node.right);
                    stack.push(&node.left);
                }
            }
        }

        Ok(())
    }
}

impl<P, V: ParameterizedDecode<P>> ParameterizedDecode<P> for Node<V> {
    fn decode_with_param(param: &P, bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut out = Node::new(V::decode_with_param(param, bytes)?);

        let mut stack = Vec::new();
        stack.push(&mut out.right);
        stack.push(&mut out.left);

        // Decode nodes in a pre-order traversal.
        while let Some(elem) = stack.pop() {
            match CodecMarker::decode(bytes)? {
                CodecMarker::Leaf => (),
                CodecMarker::Inner => {
                    *elem = Some(Box::new(Node::new(V::decode_with_param(param, bytes)?)));
                    let node = elem.as_mut().unwrap();
                    stack.push(&mut node.right);
                    stack.push(&mut node.left);
                }
            };
        }

        Ok(out)
    }
}

#[cfg(feature = "test-util")]
impl<V: core::fmt::Display> core::fmt::Display for Node<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Item<'a, V> {
            name: String,
            level: usize,
            node: Option<&'a Node<V>>,
        }

        let mut stack = vec![Item {
            name: String::default(),
            level: 0,
            node: Some(self),
        }];

        while let Some(Item { name, level, node }) = stack.pop() {
            if let Some(Node { value, left, right }) = node {
                let prefix = "  ".repeat(level);
                writeln!(f, "{name}\n{prefix}node: {value}")?;

                stack.push(Item {
                    name: format!("{prefix}right:"),
                    level: level + 1,
                    node: right.as_deref(),
                });

                stack.push(Item {
                    name: format!("{prefix}left:"),
                    level: level + 1,
                    node: left.as_deref(),
                });
            } else {
                writeln!(f, "{name} <None>")?
            };
        }

        Ok(())
    }
}

impl<V: Encode> Encode for BinaryTree<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match &self.root {
            None => CodecMarker::Leaf.encode(bytes),
            Some(node) => {
                CodecMarker::Inner.encode(bytes)?;
                node.encode(bytes)
            }
        }
    }
}

impl<P, V: ParameterizedDecode<P>> ParameterizedDecode<P> for BinaryTree<V> {
    fn decode_with_param(param: &P, bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(match CodecMarker::decode(bytes)? {
            CodecMarker::Leaf => Self::default(),
            CodecMarker::Inner => Self {
                root: Some(Box::new(Node::decode_with_param(param, bytes)?)),
            },
        })
    }
}

#[cfg(feature = "test-util")]
impl<V> core::fmt::Display for BinaryTree<V>
where
    V: core::fmt::Display,
    Node<V>: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "---")?;
        if let Some(node) = &self.root {
            writeln!(f, "{node}")?
        };
        Ok(())
    }
}

/// Marker used to distinguish between full and empty nodes.
#[repr(u8)]
#[derive(Debug)]
enum CodecMarker {
    Leaf,
    Inner,
}

impl Encode for CodecMarker {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match self {
            CodecMarker::Leaf => 0u8.encode(bytes),
            CodecMarker::Inner => 1u8.encode(bytes),
        }
    }
}

impl Decode for CodecMarker {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        match u8::decode(bytes)? {
            0u8 => Ok(CodecMarker::Leaf),
            1u8 => Ok(CodecMarker::Inner),
            _ => Err(CodecError::UnexpectedValue),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;
    use std::io::Cursor;

    use bitvec::{bits, order::Lsb0, vec::BitVec, view::BitView};
    use num_traits::Num;

    use crate::{
        bt::{BinaryTree, BinaryTreeError},
        codec::{Encode, ParameterizedDecode},
    };

    #[test]
    fn empty_tree() {
        let mut tree = BinaryTree::<u32>::default();
        assert!(tree.get(bits!()).is_none());
        assert!(tree.get_node(bits!()).is_none());
    }

    #[test]
    fn serialize_root() {
        let tree = BinaryTree::<u32>::default();
        check_serialize(tree, &());
    }

    #[test]
    fn serialize_full() {
        for size in 0..=8 {
            let prefixes = gen_prefixes(size);
            let mut tree = BinaryTree::<u32>::default();

            insert_prefixes(&mut tree, &prefixes).unwrap();
            check_serialize(tree, &());
        }
    }

    fn check_serialize<T: Encode + ParameterizedDecode<P>, P>(first: T, param: &P) {
        let bytes_first = first.get_encoded().unwrap();
        let second = T::decode_with_param(param, &mut Cursor::new(&bytes_first)).unwrap();
        let bytes_second = second.get_encoded().unwrap();

        assert_eq!(bytes_first, bytes_second);
    }

    #[test]
    fn check_full_tree() {
        for size in 0..=8 {
            let prefixes = gen_prefixes(size);
            let mut tree = BinaryTree::<u32>::default();
            insert_prefixes(&mut tree, &prefixes).unwrap();
            verify_prefixes(&tree, &prefixes)
        }
    }

    fn gen_prefixes(size: usize) -> Vec<Vec<BitVec>> {
        let mut prefixes = Vec::with_capacity(size + 1);
        for i in 0..=size {
            let num = 1usize << i;
            let mut prefixes_size_i = Vec::with_capacity(num);
            for j in 0..num {
                prefixes_size_i.push(j.view_bits::<Lsb0>()[..i].to_bitvec());
            }
            prefixes.push(prefixes_size_i);
        }

        prefixes
    }

    fn insert_prefixes<T: Num + Copy>(
        tree: &mut BinaryTree<T>,
        prefixes: &Vec<Vec<BitVec>>,
    ) -> Result<(), BinaryTreeError<T>> {
        let mut ctr = T::zero();
        for prefixes_size_i in prefixes {
            for path in prefixes_size_i {
                tree.insert(path, ctr)?;
                ctr = ctr + T::one();
            }
        }

        Ok(())
    }

    fn verify_prefixes<T: Num + Debug>(tree: &BinaryTree<T>, prefixes: &Vec<Vec<BitVec>>) {
        let mut ctr = T::zero();
        for prefixes_size_i in prefixes {
            for path in prefixes_size_i {
                let value = tree.get(path).unwrap();
                assert_eq!(*value, ctr, "path: {}", path);
                ctr = ctr + T::one();
            }
        }
    }
}
