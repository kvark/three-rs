#![warn(missing_docs)]
//! Three.js inspired 3D engine in Rust.

#[macro_use]
extern crate bitflags;
extern crate cgmath;
extern crate froggy;
extern crate genmesh;
#[macro_use]
extern crate gfx;
extern crate image;
#[macro_use]
extern crate itertools;
#[macro_use]
extern crate log;
extern crate mint;
extern crate obj;
extern crate winit;
// OpenGL
#[cfg(feature = "opengl")]
extern crate gfx_device_gl;
#[cfg(feature = "opengl")]
extern crate gfx_window_glutin;
#[cfg(feature = "opengl")]
extern crate glutin;

#[macro_use]
mod macros;
mod camera;
mod factory;
mod input;
mod render;
mod scene;
#[cfg(feature = "opengl")]
mod window;

pub use camera::{OrbitControls, Orthographic, Perspective};
pub use factory::{Factory, FilterMethod, Geometry, GeometryShape, ShadowMap, Sampler, Texture, WrapMode};
pub use input::{Button, KeyAxis, Timer, Input,
                KEY_ESCAPE, KEY_SPACE, MOUSE_LEFT, MOUSE_RIGHT,
                AXIS_LEFT_RIGHT, AXIS_DOWN_UP};
pub use render::{ColorFormat, DepthFormat, Renderer, ShadowType, DebugQuadHandle};
pub use scene::{Color, Background, Material, NodeTransform, NodeInfo,
                Group, Sprite,
                AmbientLight, DirectionalLight, HemisphereLight, PointLight};
#[cfg(feature = "opengl")]
pub use window::Window;
#[cfg(feature = "opengl")]
pub use glutin::VirtualKeyCode as Key;
pub use gfx::Primitive as GfxPrimitive;
pub use gfx::state as gfx_state;
pub use gfx::preset as gfx_preset;

use std::sync::{mpsc, Arc, Mutex};

use cgmath::Transform as Transform_;
use factory::SceneId;
use render::{DynamicData, GpuData};

/// Pointer to a Node
pub type NodePointer = froggy::Pointer<Node>;
type Transform = cgmath::Decomposed<cgmath::Vector3<f32>, cgmath::Quaternion<f32>>;

#[derive(Clone, Debug)]
enum SubLight {
    Ambient,
    Directional,
    Hemisphere { ground: Color },
    Point,
}

#[derive(Clone, Debug)]
enum ShadowProjection {
    Ortho(Orthographic),
}

#[derive(Clone, Debug)]
struct LightData {
    color: Color,
    intensity: f32,
    sub_light: SubLight,
    shadow: Option<(ShadowMap, ShadowProjection)>,
}

#[derive(Debug)]
enum SubNode {
    Empty,
    Visual(Material, GpuData),
    Light(LightData),
}

/// Fat node of the scene graph.
///
/// `Node` is used by `three-rs` internally,
/// client code uses [`Object`](struct.Object.html) instead.
#[derive(Debug)]
pub struct Node {
    visible: bool,
    world_visible: bool,
    transform: Transform,
    world_transform: Transform,
    parent: Option<NodePointer>,
    scene_id: Option<SceneId>,
    sub_node: SubNode,
}

//Note: no local state should be here, only remote links
/// `Object` represents an entity that can be added to the scene.
///
/// There is no need to use `Object` directly, there are specific wrapper types
/// for each case (e.g. [`Camera`](struct.Camera.html),
/// [`AmbientLight`](struct.AmbientLight.html),
/// [`Mesh`](struct.Mesh.html), ...).
#[derive(Clone, Debug)]
pub struct Object {
    node: NodePointer,
    tx: mpsc::Sender<Message>,
}

/// Camera is used to render Scene with specific `Projection`.
pub struct Camera<P> {
    object: Object,
    /// Projection parameters of this camera.
    pub projection: P,
}


type Message = (froggy::WeakPointer<Node>, Operation);
enum Operation {
    SetParent(NodePointer),
    SetVisible(bool),
    SetTransform(Option<mint::Point3<f32>>, Option<mint::Quaternion<f32>>, Option<f32>),
    SetMaterial(Material),
    SetTexelRange(mint::Point2<i16>, mint::Vector2<u16>),
    SetShadow(ShadowMap, ShadowProjection),
}

type HubPtr = Arc<Mutex<Hub>>;
struct Hub {
    nodes: froggy::Storage<Node>,
    message_tx: mpsc::Sender<Message>,
    message_rx: mpsc::Receiver<Message>,
}

impl Hub {
    fn new() -> HubPtr {
        let (tx, rx) = mpsc::channel();
        let hub = Hub {
            nodes: froggy::Storage::new(),
            message_tx: tx,
            message_rx: rx,
        };
        Arc::new(Mutex::new(hub))
    }

    fn process_messages(&mut self) {
        while let Ok((pnode, operation)) = self.message_rx.try_recv() {
            let node = match pnode.upgrade() {
                Ok(ptr) => &mut self.nodes[&ptr],
                Err(_) => continue,
            };
            match operation {
                Operation::SetParent(parent) => {
                    node.parent = Some(parent);
                }
                Operation::SetVisible(visible) => {
                    node.visible = visible;
                }
                Operation::SetTransform(pos, rot, scale) => {
                    if let Some(pos) = pos {
                        node.transform.disp = mint::Vector3::from(pos).into();
                    }
                    if let Some(rot) = rot {
                        node.transform.rot = rot.into();
                    }
                    if let Some(scale) = scale {
                        node.transform.scale = scale;
                    }
                }
                Operation::SetMaterial(material) => {
                    if let SubNode::Visual(ref mut mat, _) = node.sub_node {
                        *mat = material;
                    }
                }
                Operation::SetTexelRange(base, size) => {
                    if let SubNode::Visual(ref mut material, _) = node.sub_node {
                        match *material {
                            Material::Sprite { ref mut map } => map.set_texel_range(base, size),
                            _ => panic!("Unsupported material for texel range request")
                        }
                    }
                }
                Operation::SetShadow(map, proj) => {
                    if let SubNode::Light(ref mut data) = node.sub_node {
                        data.shadow = Some((map, proj));
                    }
                }
            }
        }
        self.nodes.sync_pending();
    }

    fn update_graph(&mut self) {
        let mut cursor = self.nodes.cursor_alive();
        while let Some(mut item) = cursor.next() {
            if !item.visible {
                item.world_visible = false;
                continue
            }
            let (visibility, affilation, transform) = match item.parent {
                Some(ref parent_ptr) => {
                    match item.look_back(parent_ptr) {
                        Some(parent) => {
                            (parent.world_visible,
                             parent.scene_id,
                             parent.world_transform.concat(&item.transform))
                        }
                        None => {
                            error!("Parent node was created after the child, ignoring");
                            (false, item.scene_id, item.transform)
                        }
                    }
                },
                None => (true, item.scene_id, item.transform),
            };
            item.world_visible = visibility;
            item.scene_id = affilation;
            item.world_transform = transform;
        }
    }

    fn update_mesh(&mut self, mesh: &DynamicMesh) {
        match self.nodes[&mesh.node].sub_node {
            SubNode::Visual(_, ref mut gpu_data) =>
                gpu_data.pending = Some(mesh.dynamic.clone()),
            _ => unreachable!()
        }
    }
}

/// Game scene contains game objects and can be rendered by [`Camera`](struct.Camera.html).
pub struct Scene {
    unique_id: SceneId,
    node: NodePointer,
    tx: mpsc::Sender<Message>,
    hub: HubPtr,
    /// See [`Background`](struct.Background.html).
    pub background: scene::Background,
}

/// [`Geometry`](struct.Geometry.html) with some [`Material`](struct.Material.html).
///
/// # Examples
///
/// Creating a red triangle.
///
/// ```rust,no_run
/// # let shaders_path = format!("{}/data/shaders", env!("CARGO_MANIFEST_DIR"));
/// # let mut win = three::Window::new("Example", &shaders_path).build();
/// # let factory = &mut win.factory;
/// let vertices = vec![
///     [-0.5, -0.5, 0.0].into(),
///     [ 0.5, -0.5, 0.0].into(),
///     [ 0.5, -0.5, 0.0].into(),
/// ];
/// let geometry = three::Geometry::from_vertices(vertices);
/// let red_material = three::Material::MeshBasic {
///     color: 0xFF0000,
///     wireframe: false,
///     map: None,
/// };
/// let mesh = factory.mesh(geometry, red_material);
/// # let _ = mesh;
/// ```
///
/// Duplicating a mesh.
///
/// ```rust,no_run
/// # let shaders_path = format!("{}/data/shaders", env!("CARGO_MANIFEST_DIR"));
/// # let mut win = three::Window::new("Example", &shaders_path).build();
/// # let factory = &mut win.factory;
/// # let vertices = vec![
/// #     [-0.5, -0.5, 0.0].into(),
/// #     [ 0.5, -0.5, 0.0].into(),
/// #     [ 0.5, -0.5, 0.0].into(),
/// # ];
/// # let geometry = three::Geometry::from_vertices(vertices);
/// # let red_material = three::Material::MeshBasic {
/// #     color: 0xFF0000,
/// #     wireframe: false,
/// #     map: None,
/// # };
/// # let mesh = factory.mesh(geometry, red_material);
/// let yellow_material = three::Material::MeshBasic {
///    color: 0xFFFF00,
///    wireframe: false,
///    map: None,
/// };
/// let mut duplicate = factory.mesh_instance(&mesh, Some(yellow_material));
/// // Duplicated meshes share their geometry but can be transformed individually
/// // and be rendered with different materials.
/// duplicate.set_position([1.2, 3.4, 5.6]);
/// ```
///
/// # Notes
///
/// * Meshes are removed from the scene when dropped.
/// * Hence, meshes must be kept in scope in order to be displayed.
pub struct Mesh {
    object: Object,
}

/// A dynamic version of a mesh allows changing the geometry on CPU side
/// in order to animate the mesh.
pub struct DynamicMesh {
    object: Object,
    geometry: Geometry,
    dynamic: DynamicData,
}
