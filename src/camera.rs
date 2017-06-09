use std::ops;

use cgmath;
use froggy::Pointer;
use mint;

use {Camera, Projection, Node, Object};


impl<P> AsRef<Pointer<Node>> for Camera<P> {
    fn as_ref(&self) -> &Pointer<Node> {
        &self.object.node
    }
}

impl<P> ops::Deref for Camera<P> {
    type Target = Object;
    fn deref(&self) -> &Object {
        &self.object
    }
}
impl<P> ops::DerefMut for Camera<P> {
    fn deref_mut(&mut self) -> &mut Object {
        &mut self.object
    }
}

impl Projection for cgmath::Ortho<f32> {
    fn get_matrix(&self, aspect: f32) -> mint::ColumnMatrix4<f32> {
        let center = 0.5 * (self.left + self.right);
        let offset = 0.5 * aspect * (self.top - self.bottom);
        let m: [[f32; 4]; 4] = cgmath::ortho(center - offset, center + offset,
                                             self.bottom, self.top,
                                             self.near, self.far
                                             ).into();
        m.into()
    }
}

impl Projection for cgmath::PerspectiveFov<f32> {
    fn get_matrix(&self, aspect: f32) -> mint::ColumnMatrix4<f32> {
        let m: [[f32; 4]; 4] = cgmath::perspective(self.fovy, aspect,
                                                   self.near, self.far
                                                   ).into();
        m.into()
    }
}
