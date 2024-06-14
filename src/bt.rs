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
//! and [`BinaryTree::decode`] functions, respectively.
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
#![allow(unused_variables)]

use core::fmt::Debug;
use std::io::Cursor;

use bitvec::slice::BitSlice;

use crate::codec::{CodecError, Decode, Encode};

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

/// Represents a node of a binary tree.
pub struct Node<V> {
    value: V,
    left: Option<usize>,
    right: Option<usize>,
}

/// Represents an append-only binary tree.
pub struct BinaryTree<V> {
    nodes: Vec<Node<V>>,
    root: Option<usize>,
}

impl<V> BinaryTree<V> {
    /// Number of nodes pre-allocated each time the tree grows.
    const NODES_CAPACITY: usize = 256;

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
    /// value at the end of the path.
    /// - `Err(UnreachablePath(value))` when the end of the path is unreachable.
    ///
    /// In the error cases, no insertion occurs and the value is returned to
    /// the caller of this function.
    pub fn insert(&mut self, path: &Path, value: V) -> Result<(), BinaryTreeError<V>> {
        let last = self.nodes.len();
        let mut node = &mut self.root;
        for bit in path.iter() {
            match *node {
                None => return Err(BinaryTreeError::UnreachableNode(value)),
                Some(next) => {
                    let n = &mut self.nodes[next];
                    node = if !bit { &mut n.left } else { &mut n.right };
                }
            }
        }

        if node.is_some() {
            return Err(BinaryTreeError::InsertNonEmptyNode(value));
        } else {
            *node = Some(last);

            if self.nodes.len() == self.nodes.capacity() {
                self.nodes.reserve(Self::NODES_CAPACITY);
            }

            self.nodes.push(Node {
                value,
                left: None,
                right: None,
            });
        }

        Ok(())
    }

    /// Gets a reference to the value located at the end of the path.
    ///
    /// This function traverses the tree from the root node until reaching the
    /// node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent. Otherwise, it returns a reference to the
    /// value stored in the node.
    pub fn get(&self, path: &Path) -> Option<&V> {
        let mut node = self.root;
        for bit in path {
            let next = &self.nodes[node?];
            node = if !bit { next.left } else { next.right };
        }

        Some(&self.nodes[node?].value)
    }

    /// Gets a mutable reference to the node located at the end of the path.
    ///
    /// This function traverses the tree from the root node until reaching the
    /// node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent. Otherwise, it returns a mutable reference
    /// to the node.
    pub fn get_node(&mut self, path: &Path) -> Option<&mut Node<V>> {
        let mut node = self.root;
        for bit in path {
            let next = &self.nodes[node?];
            node = if !bit { next.left } else { next.right };
        }

        Some(&mut self.nodes[node?])
    }
}

impl<V> Default for BinaryTree<V> {
    fn default() -> Self {
        Self {
            root: Option::default(),
            nodes: Vec::with_capacity(Self::NODES_CAPACITY),
        }
    }
}

// Indicates the datatype used to serialize usize values.
type UsizeType = u16;

impl Encode for Option<usize> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match *self {
            None => UsizeType::encode(&0, bytes),
            Some(n) => UsizeType::try_from(n)
                .map_err(|_| CodecError::UnexpectedValue)?
                .encode(bytes),
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        UsizeType::encoded_len(&0)
    }
}

impl Decode for Option<usize> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let value = UsizeType::decode(bytes)?.into();

        Ok((value != 0).then_some(value))
    }
}

impl<V: Encode> Encode for Node<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.value.encode(bytes)?;
        self.left.encode(bytes)?;
        self.right.encode(bytes)
    }

    fn encoded_len(&self) -> Option<usize> {
        self.value
            .encoded_len()
            .zip(self.left.encoded_len())
            .zip(self.right.encoded_len())
            .map(|((v, l), r)| v + l + r)
    }
}

impl<V: Decode> Decode for Node<V> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self {
            value: V::decode(bytes)?,
            left: Option::<usize>::decode(bytes)?,
            right: Option::<usize>::decode(bytes)?,
        })
    }
}

impl<V: Encode> Encode for BinaryTree<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        let len = self.nodes.len();
        UsizeType::try_from(len)
            .map_err(|_| CodecError::UnexpectedValue)?
            .encode(bytes)?;

        for node in self.nodes.iter() {
            node.encode(bytes)?;
        }

        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        UsizeType::encoded_len(&0)
            .zip(self.nodes.first().map_or(Some(0), Encode::encoded_len))
            .map(|(header_len, node_len)| header_len + node_len * self.nodes.len())
    }
}

impl<V: Decode> Decode for BinaryTree<V> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        let mut tree = Self::default();
        let header = UsizeType::decode(bytes)?;
        if header != 0 {
            let len = header.into();
            tree.nodes = Vec::with_capacity(len + Self::NODES_CAPACITY);
            for _ in 0..len {
                tree.nodes.push(Node::decode(bytes)?);
            }
            tree.root = Some(0);
        }

        Ok(tree)
    }
}

#[cfg(feature = "test-util")]
impl<V: core::fmt::Display> core::fmt::Display for Node<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} L: {:?} R: {:?}", self.value, self.left, self.right)
    }
}

#[cfg(feature = "test-util")]
impl<V: core::fmt::Display> core::fmt::Display for BinaryTree<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "--- Begin Tree ---")?;
        for node in self.nodes.iter() {
            writeln!(f, "{node}")?;
        }
        write!(f, "--- End Tree ---")
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
        codec::{Decode, Encode},
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
        check_serialize(tree);
    }

    #[test]
    fn serialize_full() {
        for size in 0..=8 {
            let prefixes = gen_prefixes(size);
            let mut tree = BinaryTree::<u32>::default();

            insert_prefixes(&mut tree, &prefixes).unwrap();
            check_serialize(tree);
        }
    }

    fn check_serialize<T: Encode + Decode>(first: T) {
        let bytes_first = first.get_encoded().unwrap();
        let second = T::decode(&mut Cursor::new(&bytes_first)).unwrap();
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
