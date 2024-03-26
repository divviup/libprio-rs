// SPDX-License-Identifier: MPL-2.0

//! An append-only binary tree.
//!
//! ## Properties:
//! - Binary tree: nodes have either 0, 1, or 2 child nodes.
//! - Append-only: tree only grows, no values can be removed, just queried.
//! - Serializable: tree implements Encode and Decode, iff the payload does so.
//! ## Creation
//! Use [`BinaryTree::new`] to create a binary tree, note that there must be one value
//! at the root node.
//! ## Insertion
//! Given a value and a binary path (b1, ..., bN), [`BinaryTree::insert`] will store
//! the value in the first available node along the path, unless the tree already has
//! values along the entire path.
//! ## Query
//! Given a binary path (b1, ..., bN), [`BinaryTree::get`] will return a reference to
//! the value stored exactly at the path requested. No value is returned if the tree
//! cannot be fully traversed. The empty path returns the value at the root node.
//! ## Serialization
//! A tree can be serializad to and from bytes using [`BinaryTree::encode`] and
//! [`BinaryTree::decode`] functions, respectively.
//! One-byte markers are used allowing to store sparse trees with a lower overhead.
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
//! ```
//! use prio::bt::BinaryTree;
//! use bitvec::{bits,prelude::Lsb0};
//! let mut tree = BinaryTree::new('@');
//! tree.insert(bits!(0), 'a');
//! tree.insert(bits!(1), 'b');
//! tree.insert(bits!(0, 0), 'c');
//! tree.insert(bits!(0, 1), 'd');
//! tree.insert(bits!(1, 0), 'e');
//! tree.insert(bits!(1, 1), 'f');
//! ```

use core::fmt::Debug;
use core::fmt::{Display, Formatter};
use std::io::Cursor;

use bitvec::slice::BitSlice;

use crate::codec::{CodecError, Decode, Encode};

/// Used to indicate a traversal path on the binary tree.
pub type Path = BitSlice;

/// Represents an append-only binary tree.
pub struct BinaryTree<V> {
    root: Node<V>,
}

impl<V> BinaryTree<V> {
    /// Creates a binary tree initialized with a value as the root node.
    pub fn new(value: V) -> Self {
        Self {
            root: Node::new(value),
        }
    }

    /// Inserts the value in the first available node along the path.
    /// No insertion occurs if the nodes along the path are already filled.
    pub fn insert(&mut self, path: &Path, value: V) {
        self.root.insert(path, value)
    }

    /// Gets a reference to the value exactly located at the path indicated.
    /// [`None`] is returned if no values are found during the traversal of the tree
    /// along the path.
    pub fn get(&self, path: &Path) -> Option<&V> {
        self.root.get(path)
    }
}

impl<V: Encode> Encode for BinaryTree<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        self.root.encode(bytes)
    }
}

impl<V: Decode> Decode for BinaryTree<V> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        Ok(Self {
            root: Node::decode(bytes)?,
        })
    }
}

impl<V: Display> Display for BinaryTree<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "■\n{}", self.root)
    }
}

/// Represents a node of a binary tree.
#[derive(Debug)]
struct Node<V> {
    value: V,
    left: Option<Box<Node<V>>>,
    right: Option<Box<Node<V>>>,
}

impl<V> Node<V> {
    fn new(value: V) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }

    fn insert(&mut self, path: &Path, value: V) {
        let new_node = Some(Box::new(Node::new(value)));
        let mut iter = Some(self);

        for bit in path.iter() {
            let node = iter.take().unwrap();

            if !bit {
                match node.left {
                    None => {
                        node.left = new_node;
                        break;
                    }
                    Some(_) => iter = node.left.as_deref_mut(),
                };
            } else {
                match node.right {
                    None => {
                        node.right = new_node;
                        break;
                    }
                    Some(_) => iter = node.right.as_deref_mut(),
                };
            }
        }
    }

    pub fn get(&self, path: &Path) -> Option<&V> {
        let mut iter = Some(self);

        for bit in path.iter() {
            let node = iter.unwrap();

            if !bit {
                match node.left {
                    None => return None,
                    Some(_) => iter = node.left.as_deref(),
                };
            } else {
                match node.right {
                    None => return None,
                    Some(_) => iter = node.right.as_deref(),
                };
            }
        }

        iter.map(|node| &node.value)
    }
}

impl<V: Encode> Encode for Node<V> {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        let mut stack = Vec::new();
        stack.push(Some(self));

        // Nodes are stored following a pre-order traversal.
        while let Some(elem) = stack.pop() {
            if let Some(node) = elem {
                CodecMarker::Node.encode(bytes)?;
                node.value.encode(bytes)?;
                stack.push(node.right.as_deref());
                stack.push(node.left.as_deref());
            } else {
                CodecMarker::None.encode(bytes)?;
            }
        }

        Ok(())
    }
}

impl<V: Decode> Decode for Node<V> {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        struct NodeEntry<V> {
            node: Node<V>,
            parent: Option<usize>,
            is_right: bool,
        }

        struct Item {
            is_right: bool,
            index: usize,
        }

        let root = match CodecMarker::decode(bytes)? {
            CodecMarker::Node => Node::new(V::decode(bytes)?),
            _ => return Err(CodecError::UnexpectedValue),
        };

        let mut list_nodes = Vec::new();
        list_nodes.push(NodeEntry {
            node: root,
            parent: None,
            is_right: false,
        });

        let mut stack = Vec::new();
        stack.push(Item {
            index: 0,
            is_right: true,
        });
        stack.push(Item {
            index: 0,
            is_right: false,
        });

        // Decode nodes in a pre-order traversal and store them in a list.
        while let Some(elem) = stack.pop() {
            match CodecMarker::decode(bytes)? {
                CodecMarker::None => (),
                CodecMarker::Node => {
                    let index = list_nodes.len();
                    list_nodes.push(NodeEntry {
                        node: Node::new(V::decode(bytes)?),
                        parent: Some(elem.index),
                        is_right: elem.is_right,
                    });
                    stack.push(Item {
                        is_right: true,
                        index,
                    });
                    stack.push(Item {
                        is_right: false,
                        index,
                    });
                }
            };
        }

        // Each node is assigned to its parent.
        let mut out = None;
        while let Some(elem) = list_nodes.pop() {
            if let Some(parent) = elem.parent {
                let new_node = Some(Box::new(elem.node));
                if elem.is_right {
                    list_nodes[parent].node.right = new_node
                } else {
                    list_nodes[parent].node.left = new_node
                }
            } else {
                out = Some(elem.node);
            }
        }

        out.ok_or(CodecError::Other("failed to decode node".into()))
    }
}

/// Marker used to distinguish between full and empty nodes.
#[repr(u8)]
#[derive(Debug)]
enum CodecMarker {
    None,
    Node,
}

impl Encode for CodecMarker {
    fn encode(&self, bytes: &mut Vec<u8>) -> Result<(), CodecError> {
        match self {
            CodecMarker::None => 0u8.encode(bytes),
            CodecMarker::Node => 1u8.encode(bytes),
        }
    }
}

impl Decode for CodecMarker {
    fn decode(bytes: &mut Cursor<&[u8]>) -> Result<Self, CodecError> {
        match u8::decode(bytes)? {
            0u8 => Ok(CodecMarker::None),
            1u8 => Ok(CodecMarker::Node),
            _ => Err(CodecError::UnexpectedValue),
        }
    }
}

impl<V: Display> Display for Node<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        struct Item<'a, V> {
            name: String,
            level: usize,
            value: Option<&'a Node<V>>,
        }

        let mut stack = Vec::new();

        stack.push(Item {
            name: String::default(),
            level: 0,
            value: Some(self),
        });

        while let Some(mut elem) = stack.pop() {
            if let Some(node) = elem.value {
                let prefix = "  ".repeat(elem.level);

                if !elem.name.is_empty() {
                    elem.name.push('\n')
                }

                writeln!(f, "{}{}└node: {}", elem.name, prefix, node.value)?;

                stack.push(Item {
                    name: format!("{}└right:", prefix),
                    level: elem.level + 1,
                    value: node.right.as_deref(),
                });

                stack.push(Item {
                    name: format!("{}└left:", prefix),
                    level: elem.level + 1,
                    value: node.left.as_deref(),
                });
            } else {
                writeln!(f, "{} <None>", elem.name)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use core::fmt::Debug;
    use std::io::Cursor;

    use bitvec::{order::Lsb0, vec::BitVec, view::BitView};
    use num_traits::Num;

    use crate::{
        bt::BinaryTree,
        codec::{Decode, Encode},
    };

    #[test]
    fn serialize_root() {
        let tree = BinaryTree::new(u64::MAX);
        check_serialize(tree);
    }

    #[test]
    fn serialize_full() {
        for level in 0..=8 {
            let prefixes = gen_prefixes(level);
            let mut tree = BinaryTree::new(0u32);

            insert_prefixes(&mut tree, &prefixes);
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
        for level in 0..=8 {
            let prefixes = gen_prefixes(level);
            let mut tree = BinaryTree::new(0u32);
            insert_prefixes(&mut tree, &prefixes);
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

    fn insert_prefixes<T: Num + Copy>(tree: &mut BinaryTree<T>, prefixes: &Vec<Vec<BitVec>>) {
        let mut ctr = T::zero();
        for prefixes_size_i in prefixes {
            for path in prefixes_size_i {
                tree.insert(path, ctr);
                ctr = ctr + T::one();
            }
        }
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
