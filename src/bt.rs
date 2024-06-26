// SPDX-License-Identifier: MPL-2.0

//! An append-only binary tree.
//!
//! ## Properties:
//! - Binary tree: nodes have either 0, 1, or 2 child nodes.
//! - Append-only: tree only grows, values can only be inserted and queried.
//! - Serializable: tree can be coverted to and from bytes.
//!
//! ## Creation
//! Use [`BinaryTree::default`] to create a binary tree.
//!
//! ## Insertion
//! Given a value and a binary path, [`BinaryTree::insert`] traverses the
//! tree along the path starting from the root and inserts the value at the
//! end of the path. [`BinaryTree::insert_at`] additionally takes a node
//! reference that indicates the start of the traversal. No insertion occurs
//! when the end of the path is unreachable or when the tree already contains
//! a value at the specified location.
//!
//! ## Query
//! Given a binary path, [`BinaryTree::get_node`] traverses the tree along
//! the path starting from the root and returns a reference to the node
//! located at the end of the path. No value is returned if the end of the
//! path is unreachable. [`BinaryTree::get_value`] behaves similarly but
//! instead returns a reference to the value stored in the node.
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
//! ```txt
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
//! ```

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
    /// Error when a node reference is invalid.
    #[error("invalid reference to a node in the tree")]
    InvalidNodeRef(V),
}

/// Used to indicate a traversal path on the binary tree.
pub type Path = BitSlice;

/// Represents a node of a binary tree.
struct Node<V> {
    value: V,
    left: Option<usize>,
    right: Option<usize>,
}

impl<V> Node<V> {
    fn new(value: V) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }
}

/// NodeRef is a reference to a node of the tree.
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct NodeRef(usize);

impl NodeRef {
    const ROOT: NodeRef = NodeRef(0);
}

/// Represents an append-only binary tree.
pub struct BinaryTree<V> {
    nodes: Vec<Node<V>>,
}

impl<V> BinaryTree<V> {
    /// Inserts the value at the location reached by traversing the tree
    /// along the path starting from the root node.
    ///
    /// This function traverses the tree from the root until reaching the
    /// node at the end of the path. If the node is unreachable or already
    /// contains a value, the function returns an error wrapping the value;
    /// otherwise, it stores the value in a new node and returns a reference
    /// to the node.
    ///
    /// # Returns
    /// - `Ok(NodeRef)` a reference is returned when the node is inserted
    /// at the end of the path.
    /// - `Err(InsertNonEmptyNode(value))` when the tree already contains a
    /// value at the end of the path.
    /// - `Err(UnreachableNode(value))` when the end of the path is unreachable.
    ///
    /// In the error cases, no insertion occurs and the value is returned to
    /// the caller of this function.
    pub fn insert(&mut self, path: &Path, value: V) -> Result<NodeRef, BinaryTreeError<V>> {
        self.insert_at(NodeRef::ROOT, path, value)
    }

    /// Inserts the value at the location reached by traversing the tree
    /// along the path starting from the specified node reference.
    ///
    /// This function traverses the tree from the node specified until
    /// reaching the node at the end of the path. If the node is unreachable
    /// or already contains a value, the function returns an error wrapping
    /// the value; otherwise, it stores the value in a new node and returns
    /// a reference to the node.
    ///
    /// # Returns
    /// - `Ok(NodeRef)` a reference is returned when the node is inserted
    /// at the end of the path.
    /// - `Err(InsertNonEmptyNode(value))` when the tree already contains a
    /// value at the end of the path.
    /// - `Err(UnreachableNode(value))` when the end of the path is unreachable.
    /// - `Err(InvalidNodeRef(value))` when the node reference is invalid.
    ///
    /// In the error cases, no insertion occurs and the value is returned to
    /// the caller of this function.
    pub fn insert_at(
        &mut self,
        node_ref: NodeRef,
        path: &Path,
        value: V,
    ) -> Result<NodeRef, BinaryTreeError<V>> {
        if !(self.is_valid_node_ref(node_ref)
            || (self.nodes.is_empty() && node_ref == NodeRef::ROOT))
        {
            return Err(BinaryTreeError::InvalidNodeRef(value));
        }

        let new_index = self.nodes.len();
        let mut node = &mut (!self.nodes.is_empty()).then_some(node_ref.0);
        for bit in path.iter() {
            match *node {
                None => return Err(BinaryTreeError::UnreachableNode(value)),
                Some(next) => {
                    let n = &mut self.nodes[next];
                    node = if !bit { &mut n.left } else { &mut n.right }
                }
            }
        }

        if node.is_some() {
            return Err(BinaryTreeError::InsertNonEmptyNode(value));
        }

        *node = Some(new_index);
        self.nodes.push(Node::new(value));

        Ok(NodeRef(new_index))
    }

    /// Gets a reference to the value stored in the node located at the end
    /// of the path traversing the tree from the root node.
    ///
    /// This function traverses the tree from the root node until reaching
    /// the node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent; otherwise, it returns a reference to the
    /// value stored in the node.
    pub fn get_value(&self, path: &Path) -> Option<&V> {
        let NodeRef(node) = self.get_node_at(NodeRef::ROOT, path)?;
        Some(&self.nodes[node].value)
    }

    /// Gets a reference to the node located at the end of the path
    /// traversing the tree from the root node.
    ///
    /// This function traverses the tree from the root node until reaching
    /// the node at the end of the path. It returns [None], if the node is
    /// unreachable or nonexistent; otherwise, it returns a reference to the
    /// node.
    pub fn get_node(&self, path: &Path) -> Option<NodeRef> {
        self.get_node_at(NodeRef::ROOT, path)
    }

    /// Gets a reference to the node located at the end of the path
    /// traversing the tree from the specified node reference.
    ///
    /// This function traverses the tree from the specified node until
    /// reaching the node at the end of the path. It returns [None],
    /// if the node is unreachable or nonexistent; otherwise, it returns
    /// a reference to the node.
    pub fn get_node_at(&self, node_ref: NodeRef, path: &Path) -> Option<NodeRef> {
        self.is_valid_node_ref(node_ref).then_some(())?;

        let mut node = (!self.nodes.is_empty()).then_some(node_ref.0);
        for bit in path.iter() {
            let next = &self.nodes[node?];
            node = if !bit { next.left } else { next.right };
        }

        node.map(NodeRef)
    }

    /// Checks whether the node reference is valid with respect to the tree.
    fn is_valid_node_ref(&self, node_ref: NodeRef) -> bool {
        node_ref.0 < self.nodes.len()
    }
}

impl<V> Default for BinaryTree<V> {
    fn default() -> Self {
        // Number of pre-allocated nodes at tree creation.
        const NODES_CAPACITY: usize = 256;
        Self {
            nodes: Vec::with_capacity(NODES_CAPACITY),
        }
    }
}

impl<V: Encode> Encode for BinaryTree<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        let mut stack = vec![(!self.nodes.is_empty()).then_some(NodeRef::ROOT.0)];

        // Nodes are stored following a pre-order traversal.
        while let Some(item) = stack.pop() {
            match item {
                None => NodeMarker::Leaf.encode(bytes)?,
                Some(next) => {
                    NodeMarker::Inner.encode(bytes)?;
                    let node = &self.nodes[next];
                    node.value.encode(bytes)?;
                    stack.push(node.right);
                    stack.push(node.left);
                }
            }
        }

        Ok(())
    }

    fn encoded_len(&self) -> Option<usize> {
        let leaf_len = NodeMarker::Leaf.encoded_len()?;
        let mut len = if self.nodes.is_empty() {
            leaf_len
        } else {
            self.nodes.len() * NodeMarker::Inner.encoded_len()?
        };

        for node in self.nodes.iter() {
            len += node.value.encoded_len()?;
            if node.left.is_none() {
                len += leaf_len;
            }
            if node.right.is_none() {
                len += leaf_len;
            }
        }

        Some(len)
    }
}

impl<V: Decode> Decode for BinaryTree<V> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        #[derive(Default)]
        struct Item {
            parent_index: Option<usize>,
            is_left: bool,
        }

        let mut tree = Self::default();
        let mut stack = vec![Item::default()];

        // Decode nodes in a pre-order traversal.
        while let Some(item) = stack.pop() {
            match NodeMarker::decode(bytes)? {
                NodeMarker::Leaf => (),
                NodeMarker::Inner => {
                    let parent_index = Some(tree.nodes.len());
                    tree.nodes.push(Node::new(V::decode(bytes)?));

                    stack.push(Item {
                        is_left: false,
                        parent_index,
                    });
                    stack.push(Item {
                        is_left: true,
                        parent_index,
                    });

                    let Some(node) = item.parent_index else { continue };
                    if item.is_left {
                        tree.nodes[node].left = parent_index;
                    } else {
                        tree.nodes[node].right = parent_index;
                    }
                }
            }
        }

        Ok(tree)
    }
}

/// Marker used to distinguish between full and empty nodes.
#[repr(u8)]
enum NodeMarker {
    Leaf,
    Inner,
}

impl Encode for NodeMarker {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match self {
            NodeMarker::Leaf => 0u8.encode(bytes),
            NodeMarker::Inner => 1u8.encode(bytes),
        }
    }

    fn encoded_len(&self) -> Option<usize> {
        u8::encoded_len(&0)
    }
}

impl Decode for NodeMarker {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        match u8::decode(bytes)? {
            0u8 => Ok(NodeMarker::Leaf),
            1u8 => Ok(NodeMarker::Inner),
            _ => Err(CodecError::UnexpectedValue),
        }
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
        bt::{BinaryTree, BinaryTreeError, NodeRef},
        codec::{Decode, Encode},
    };

    #[test]
    fn empty_tree() {
        let tree = BinaryTree::<u32>::default();
        assert!(tree.get_value(bits!()).is_none());
        assert!(tree.get_node(bits!()).is_none());
        assert!(tree.nodes.is_empty());
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
        assert_eq!(Some(bytes_first.len()), first.encoded_len());
    }

    #[test]
    fn check_full_tree() {
        for size in 0..=8 {
            let prefixes = gen_prefixes(size);
            let mut tree = BinaryTree::<u32>::default();
            insert_prefixes(&mut tree, &prefixes).unwrap();
            verify_prefixes(&tree, &prefixes);
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
                let value = tree.get_value(path).unwrap();
                assert_eq!(*value, ctr, "path: {}", path);
                ctr = ctr + T::one();
            }
        }
    }

    #[test]
    fn is_valid_node_ref() {
        let mut tree = BinaryTree::<u32>::default();
        assert!(!tree.is_valid_node_ref(NodeRef(0)));
        assert!(!tree.is_valid_node_ref(NodeRef(1)));
        assert!(tree.get_node_at(NodeRef(0), bits!()).is_none());
        assert!(tree.get_node_at(NodeRef(1), bits!()).is_none());

        let node_ref = tree.insert(bits!(), 1111).unwrap();
        assert!(tree.is_valid_node_ref(node_ref));
        assert!(tree.is_valid_node_ref(NodeRef(0)));
        assert!(!tree.is_valid_node_ref(NodeRef(1)));
        assert_eq!(tree.get_node_at(node_ref, bits!()), Some(NodeRef::ROOT));
        assert_eq!(tree.get_node_at(NodeRef(0), bits!()), Some(NodeRef::ROOT));
        assert_eq!(tree.get_node_at(NodeRef(1), bits!()), None);
    }

    #[test]
    fn insert_at() {
        let mut t0 = BinaryTree::<u32>::default();
        t0.insert(bits!(), 1).unwrap();
        t0.insert(bits!(1), 10).unwrap();
        t0.insert(bits!(1, 1), 100).unwrap();
        t0.insert(bits!(1, 1, 1), 1000).unwrap();
        t0.insert(bits!(1, 1, 1, 1), 10000).unwrap();

        let mut t1 = BinaryTree::<u32>::default();
        let mut node_ref = NodeRef::ROOT;
        node_ref = t1.insert_at(node_ref, bits!(), 1).unwrap();
        node_ref = t1.insert_at(node_ref, bits!(1), 10).unwrap();
        node_ref = t1.insert_at(node_ref, bits!(1), 100).unwrap();
        node_ref = t1.insert_at(node_ref, bits!(1), 1000).unwrap();
        node_ref = t1.insert_at(node_ref, bits!(1), 10000).unwrap();

        assert_eq!(node_ref, NodeRef(4));
        assert_eq!(t0.get_encoded().unwrap(), t1.get_encoded().unwrap());
    }

    #[test]
    fn insert_at_invalid_node_ref() {
        // `insert_at` does not accept invalid node references, when the tree is empty.
        let mut tree = BinaryTree::<u32>::default();
        let invalid_node_ref = NodeRef(77);
        assert!(!tree.is_valid_node_ref(invalid_node_ref));
        assert_eq!(
            tree.insert_at(invalid_node_ref, bits!(), 1)
                .unwrap_err()
                .to_string(),
            BinaryTreeError::InvalidNodeRef(1).to_string()
        );
    }

    #[test]
    fn insert_at_root_ok() {
        // `insert_at` does accept `NodeRef::ROOT`, when the tree is empty.
        let mut tree = BinaryTree::<u32>::default();
        let invalid_node_ref = NodeRef::ROOT;
        assert!(!tree.is_valid_node_ref(invalid_node_ref));
        assert_eq!(
            tree.insert_at(invalid_node_ref, bits!(), 1).unwrap(),
            NodeRef::ROOT
        );
    }
}
