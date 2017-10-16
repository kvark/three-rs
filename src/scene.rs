
use color::Color;
use hub::{HubPtr, Message, Operation};
use node::NodePointer;
use std::sync::mpsc;
use texture::{CubeMap, Texture};

pub type SceneId = usize;

/// Background type.
#[derive(Clone, Debug)]
pub enum Background {
    /// Basic solid color background.
    Color(Color),
    /// Texture background, covers the whole screen.
    // TODO: different wrap modes?
    Texture(Texture<[f32; 4]>),
    /// Skybox
    Skybox(CubeMap<[f32; 4]>),
}

/// Game scene contains game objects and can be rendered by [`Camera`](struct.Camera.html).
pub struct Scene {
    pub(crate) unique_id: SceneId,
    pub(crate) node: NodePointer,
    pub(crate) tx: mpsc::Sender<Message>,
    pub(crate) hub: HubPtr,
    /// See [`Background`](struct.Background.html).
    pub background: Background,
}

impl Scene {
    /// Add new [`Object`](struct.Object.html) to the scene.
    pub fn add<P: AsRef<NodePointer>>(
        &mut self,
        child: &P,
    ) {
        let msg = Operation::SetParent(self.node.clone());
        let _ = self.tx.send((child.as_ref().downgrade(), msg));
    }
}
