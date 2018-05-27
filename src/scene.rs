//! `Scene` and `SyncGuard` structures.

use node;
use color::Color;
use hub::{Hub, HubPtr, SubNode};
use object::{Base, DowncastObject, Group, Object};
use texture::{CubeMap, Texture};

use std::mem;
use std::marker::PhantomData;
use std::sync::MutexGuard;


/// Background type.
#[derive(Clone, Debug, PartialEq)]
pub enum Background {
    /// Basic solid color background.
    Color(Color),
    /// Texture background, covers the whole screen.
    // TODO: different wrap modes?
    Texture(Texture<[f32; 4]>),
    /// Skybox
    Skybox(CubeMap<[f32; 4]>),
}

/// The root node of a tree of game objects that may be rendered by a [`Camera`].
///
/// [`Camera`]: ../camera/struct.Camera.html
pub struct Scene {
    pub(crate) hub: HubPtr,
    pub(crate) first_child: Option<node::NodePointer>,
    /// See [`Background`](struct.Background.html).
    pub background: Background,
}

impl Scene {
    /// Add new [`Base`](struct.Base.html) to the scene.
    pub fn add<P>(
        &mut self,
        child_base: P,
    ) where
        P: AsRef<Base>,
    {
        let mut hub = self.hub.lock().unwrap();
        let node_ptr = child_base.as_ref().node.clone();
        let child = &mut hub[child_base];

        if child.next_sibling.is_some() {
            error!("Element {:?} is added to a scene while still having old parent - {}",
                child.sub_node, "discarding siblings");
        }

        child.next_sibling = mem::replace(&mut self.first_child, Some(node_ptr));
    }

    /// Remove a previously added [`Base`](struct.Base.html) from the scene.
    pub fn remove<P>(
        &mut self,
        child_base: P,
    ) where
        P: AsRef<Base>,
    {
        let target_maybe = Some(child_base.as_ref().node.clone());
        let mut hub = self.hub.lock().unwrap();
        let next_sibling = hub[child_base].next_sibling.clone();

        if self.first_child == target_maybe {
            self.first_child = next_sibling;
            return;
        }

        let mut cur_ptr = self.first_child.clone();
        while let Some(ptr) = cur_ptr.take() {
            let node = &mut hub.nodes[&ptr];
            if node.next_sibling == target_maybe {
                node.next_sibling = next_sibling;
                return;
            }
            cur_ptr = node.next_sibling.clone(); //TODO: avoid clone
        }

        error!("Unable to find child for removal");
    }
}


/// `SyncGuard` is used to obtain information about scene nodes in the most effective way.
///
/// # Examples
///
/// Imagine that you have your own helper type `Enemy`:
///
/// ```rust
/// # extern crate three;
/// struct Enemy {
///     mesh: three::Mesh,
///     is_visible: bool,
/// }
/// # fn main() {}
/// ```
///
/// You need this wrapper around `three::Mesh` to cache some information - in our case, visibility.
///
/// In your game you contain all your enemy objects in `Vec<Enemy>`. In the main loop you need
/// to iterate over all the enemies and make them visible or not, basing on current position.
/// The most obvious way is to use [`object::Base::sync`], but it's not the best idea from the side of
/// performance. Instead, you can create `SyncGuard` and use its `resolve` method to effectively
/// walk through every enemy in your game:
///
/// ```rust,no_run
/// # extern crate three;
/// # #[derive(Clone)]
/// # struct Enemy {
/// #     mesh: three::Mesh,
/// #     is_visible: bool,
/// # }
/// #
/// # impl three::Object for Enemy {}
/// #
/// # impl AsRef<three::object::Base> for Enemy {
/// #     fn as_ref(&self) -> &three::object::Base {
/// #         self.mesh.as_ref()
/// #     }
/// # }
/// #
/// # fn main() {
/// # use three::Object;
/// # let mut win = three::Window::new("SyncGuard example");
/// # let geometry = three::Geometry::default();
/// # let material = three::material::Basic { color: three::color::RED, map: None };
/// # let mesh = win.factory.mesh(geometry, material);
/// # let enemy = Enemy { mesh, is_visible: true };
/// # win.scene.add(&enemy);
/// # let mut enemies = vec![enemy];
/// # loop {
/// let mut sync = win.scene.sync_guard();
/// for mut enemy in &mut enemies {
///     let node = sync.resolve(enemy);
///     let position = node.transform.position;
///     if position.x > 10.0 {
///         enemy.is_visible = false;
///         enemy.set_visible(false);
///     } else {
///         enemy.is_visible = true;
///         enemy.set_visible(true);
///     }
/// }
/// # }}
/// ```
///
/// [`object::Base::sync`]: ../object/struct.Base.html#method.sync
pub struct SyncGuard<'a> {
    pub(crate) scene: &'a Scene,
    pub(crate) hub: MutexGuard<'a, Hub>,
}

impl<'a> SyncGuard<'a> {
    /// Obtains `objects`'s local space [`Node`] in an effective way.
    ///
    /// # Panics
    /// Panics if `scene` doesn't have this `object::Base`.
    ///
    /// [`Node`]: ../node/struct.Node.html
    pub fn resolve<T: 'a + Object>(
        &self,
        object: &T,
    ) -> node::Node<node::Local> {
        self.hub[object].to_node()
    }

    /// Obtains `objects`'s world [`Node`] by traversing the scene graph.
    /// *Note*: this can be slow.
    ///
    /// # Panics
    /// Panics if the doesn't have this `object::Base`.
    ///
    /// [`Node`]: ../node/struct.Node.html
    pub fn resolve_world<T: 'a + Object>(
        &self,
        object: &T,
    ) -> node::Node<node::World> {
        let internal = &self.hub[object] as *const _;
        let wn = self.hub
            .walk_all(&self.scene.first_child)
            .find(|wn| wn.node as *const _ == internal)
            .expect("Unable to find objects for world resolve!");
        node::Node {
            visible: wn.world_visible,
            name: wn.node.name.clone(),
            transform: wn.world_transform.into(),
            material: match wn.node.sub_node {
                SubNode::Visual(ref mat, _, _) => Some(mat.clone()),
                _ => None,
            },
            _space: PhantomData,
        }
    }

    /// Obtains `object`'s internal data.
    ///
    /// Three-rs objects normally expose a write-only interface, making it possible to change
    /// an object's internal values but not possible to read those values. `SyncGuard` allows
    /// for that data to be read in a controlled way.
    ///
    /// Each object type has its own internal data, and not all object types can provide access
    /// to meaningful data. The following types provide specific data you can use:
    ///
    /// * [`Base`]:
    pub fn resolve_data<T: 'a + Object>(
        &self,
        object: &T,
    ) -> T::Data {
        object.resolve_data(self)
    }

    /// Finds a node in a group, or any of its children, by name.
    ///
    /// Performs a depth-first search starting with `root` looking for an object with `name`.
    /// Returns the [`Base`] for the first object found with a matching name, otherwise returns
    /// `None` if no such object is found. Note that if more than one object exists in the
    /// hierarchy, then only the first one discovered will be returned.
    pub fn find_child_by_name(&mut self, root: &Group, name: &str) -> Option<Base> {
        let root = root.as_ref().node.clone();
        self
            .hub
            .walk_all(&Some(root))
            .find(|walked| walked.node.name.as_ref().map(|node_name| node_name == name).unwrap_or(false))
            .map(|walked| self.hub.upgrade_ptr(walked.node_ptr.clone()))
    }

    /// Returns an iterator that walks all the children
    pub fn find_children_by_name(&'a self, root: &Group, name: &'a str) -> impl Iterator<Item = Base> + 'a {
        let root = root.as_ref().node.clone();
        let hub = &self.hub;
        self
            .hub
            .walk_all(&Some(root))
            .filter(move |walked| walked.node.name.as_ref().map(|node_name| node_name == name).unwrap_or(false))
            .map(move |walked| hub.upgrade_ptr(walked.node_ptr.clone()))
    }

    /// Attempts to find an object of type `T` in the group or any of its children.
    pub fn find_child_of_type_by_name<T: DowncastObject>(&'a self, root: &Group, name: &'a str) -> Option<T> {
        self.find_children_of_type_by_name(root, name).next()
    }

    /// Returns an iterator yielding all children of `root` of type `T` named `name`.
    pub fn find_children_of_type_by_name<T: DowncastObject>(&'a self, root: &Group, name: &'a str) -> impl Iterator<Item = T> + 'a {
        let guard = &*self;
        self
            .find_children_by_name(root, name)
            .filter_map(move |base| guard.downcast(&base))
    }

    /// Returns an iterator yielding all children of `root` of type `T`.
    pub fn find_children_of_type<T: DowncastObject>(&'a self, root: &Group) -> impl Iterator<Item = T> + 'a {
        let root = root.as_ref().node.clone();
        let guard = &*self;
        self
            .hub
            .walk_all(&Some(root))
            .map(move |walked| guard.hub.upgrade_ptr(walked.node_ptr.clone()))
            .filter_map(move |base| guard.downcast(&base))
    }

    /// Returns the first child of `root` of type `T`.
    pub fn find_child_of_type<T: DowncastObject>(&'a self, root: &Group) -> Option<T> {
        self.find_children_of_type::<T>(root).next()
    }

    /// Attempts to downcast a [`Base`] to its concrete object type.
    ///
    /// If the downcast succeeds, the concrete object is returned. Returns `None` if the
    /// downcast fails.
    pub fn downcast<T: DowncastObject>(&self, base: &Base) -> Option<T> {
        let object_type = self.resolve_data(base);
        T::downcast(object_type)
    }
}

impl Scene {
    /// Create new [`SyncGuard`](struct.SyncGuard.html).
    ///
    /// This is performance-costly operation, you should not use it many times per frame.
    pub fn sync_guard(&mut self) -> SyncGuard {
        let mut hub = self.hub.lock().unwrap();
        hub.process_messages();
        SyncGuard { scene: self, hub }
    }
}
