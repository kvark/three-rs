//! Material parameters for mesh rendering.

use color;

use color::Color;
use render::BasicPipelineState;
use texture::Texture;
use util;

#[doc(inline)]
pub use self::basic::Basic;

/// Basic material API.
pub mod basic {
    use super::*;

    /// Parameters for a basic solid mesh material.
    ///
    /// Renders triangle meshes with a solid color or texture.
    #[derive(Clone, Hash, Debug, PartialEq, Eq)]
    pub struct Basic {
        /// Solid color applied in the absence of `map`.
        ///
        /// Default: `WHITE`.
        pub color: Color,

        /// Texture applied using the mesh texture co-ordinates.
        ///
        /// Default: `None`.
        pub map: Option<Texture<[f32; 4]>>,
    }

    impl Default for Basic {
        fn default() -> Self {
            Self {
                color: color::WHITE,
                map: None,
            }
        }
    }

    /// Parameters for a basic solid mesh material with a custom pipeline.
    ///
    /// Renders triangle meshes with a custom pipeline with a basic material as
    /// its input.
    #[derive(Clone, Debug, PartialEq, Hash)]
    pub struct Custom {
        /// Solid color applied in the absense of `map`.
        pub color: Color,

        /// Texture applied using the mesh texture co-ordinates.
        pub map: Option<Texture<[f32; 4]>>,

        /// The custom pipeline state object to be applied to the mesh.
        pub pipeline: BasicPipelineState,
    }

    impl Eq for Custom {}
}

/// Parameters for a Lamberian diffusion reflection model.
///
/// Renders triangle meshes with the Gouraud illumination model.
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct Lambert {
    /// Solid color applied in the absense of `map`.
    ///
    /// Default: `WHITE`.
    pub color: Color,

    /// Specifies whether lighting should be constant over faces.
    ///
    /// Default: `false` (lighting is interpolated across faces).
    pub flat: bool,
}

impl Default for Lambert {
    fn default() -> Self {
        Self {
            color: color::WHITE,
            flat: false,
        }
    }
}

/// Parameters for a line material.
///
/// Renders line strip meshes with a solid color and unit width.
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct Line {
    /// Solid line color.
    ///
    /// Default: `0xFFFFFF` (white).
    pub color: Color,
}

impl Default for Line {
    fn default() -> Self {
        Self {
            color: color::WHITE,
        }
    }
}

/// Parameters for a PBR (physically based rendering) lighting model.
///
/// Renders triangle meshes with a PBR (physically-based rendering)
/// illumination model
#[derive(Derivative)]
#[derivative(Clone, Debug, PartialEq, Hash, Eq)]
pub struct Pbr {
    /// Solid base color applied in the absence of `base_color_map`.
    ///
    /// Default: `WHITE`.
    pub base_color_factor: Color,

    /// Base color alpha factor applied in the absence of `base_color_map`.
    ///
    /// Default: `1.0` (opaque).
    #[derivative(Hash(hash_with = "util::hash_f32"))]
    pub base_color_alpha: f32,

    /// Metallic factor in the range [0.0, 1.0].
    ///
    /// Default: `1.0`.
    #[derivative(Hash(hash_with = "util::hash_f32"))]
    pub metallic_factor: f32,

    /// Roughness factor in the range [0.0, 1.0].
    ///
    /// * A value of 1.0 means the material is completely rough.
    /// * A value of 0.0 means the material is completely smooth.
    ///
    /// Default: `1.0`.
    #[derivative(Hash(hash_with = "util::hash_f32"))]
    pub roughness_factor: f32,

    /// Scalar multiplier in the range [0.0, 1.0] that controls the amount of
    /// occlusion applied in the presence of `occlusion_map`.
    ///
    /// Default: `1.0`.
    #[derivative(Hash(hash_with = "util::hash_f32"))]
    pub occlusion_strength: f32,

    /// Solid emissive color applied in the absence of `emissive_map`.
    ///
    /// Default: `BLACK`.
    pub emissive_factor: Color,

    /// Scalar multiplier applied to each normal vector of the `normal_map`.
    ///
    /// This value is ignored in the absence of `normal_map`.
    ///
    /// Default: `1.0`.
    #[derivative(Hash(hash_with = "util::hash_f32"))]
    pub normal_scale: f32,

    /// Base color texture.
    ///
    /// Default: `None`.
    pub base_color_map: Option<Texture<[f32; 4]>>,

    /// Normal texture.
    ///
    /// Default: `None`.
    pub normal_map: Option<Texture<[f32; 4]>>,

    /// Emissive texture.
    ///
    /// Default: `None`.
    pub emissive_map: Option<Texture<[f32; 4]>>,

    /// Metallic-roughness texture.
    ///
    /// Default: `None`.
    pub metallic_roughness_map: Option<Texture<[f32; 4]>>,

    /// Occlusion texture.
    ///
    /// Default: `None`.
    pub occlusion_map: Option<Texture<[f32; 4]>>,
}

impl Default for Pbr {
    fn default() -> Self {
        Self {
            base_color_factor: color::WHITE,
            base_color_alpha: 1.0,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            occlusion_strength: 1.0,
            emissive_factor: color::BLACK,
            normal_scale: 1.0,
            base_color_map: None,
            normal_map: None,
            emissive_map: None,
            metallic_roughness_map: None,
            occlusion_map: None,
        }
    }
}

/// Parameters for a Phong reflection model.
///
/// Renders triangle meshes with the Phong illumination model.
#[derive(Derivative)]
#[derivative(Clone, Debug, PartialEq, Hash, Eq)]
pub struct Phong {
    /// Solid color applied in the absense of `map`.
    ///
    /// Default: `WHITE`.
    pub color: Color,

    /// Determines the sharpness of specular highlights.
    ///
    /// Higher values result in sharper highlights to produce a glossy effect.
    ///
    /// Default: `30.0`.
    #[derivative(Hash(hash_with = "util::hash_f32"))]
    pub glossiness: f32,
}

impl Default for Phong {
    fn default() -> Self {
        Self {
            color: color::WHITE,
            glossiness: 30.0,
        }
    }
}

/// Texture for a 2D sprite.
///
/// Renders [`Sprite`] objects with the given texture.
///
/// [`Sprite`]: ../sprite/struct.Sprite.html
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct Sprite {
    /// The texture the apply to the sprite.
    pub map: Texture<[f32; 4]>,
}

/// Parameters for mesh wireframe rasterization.
///
/// Renders the edges of a triangle mesh with a solid color.
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct Wireframe {
    /// Solid color applied to each wireframe edge.
    ///
    /// Default: `WHITE`.
    pub color: Color,
}

/// Specifies the appearance of a [`Mesh`](struct.Mesh.html).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Material {
    /// Renders triangle meshes with a solid color or texture.
    Basic(Basic),

    /// Renders triangle meshes with a custom pipeline with a basic material as
    /// its input.
    CustomBasic(basic::Custom),

    /// Renders line strip meshes with a solid color and unit width.
    Line(Line),

    /// Renders triangle meshes with the Gouraud illumination model.
    Lambert(Lambert),

    /// Renders triangle meshes with the Phong illumination model.
    Phong(Phong),

    /// Renders triangle meshes with a PBR (physically-based rendering)
    /// illumination model
    Pbr(Pbr),

    /// Renders [`Sprite`] objects with the given texture.
    ///
    /// [`Sprite`]: ../sprite/struct.Sprite.html
    Sprite(Sprite),

    /// Renders the edges of a triangle mesh with a solid color.
    Wireframe(Wireframe),
}

impl From<Basic> for Material {
    fn from(params: Basic) -> Self {
        Material::Basic(params)
    }
}

impl From<basic::Custom> for Material {
    fn from(params: basic::Custom) -> Self {
        Material::CustomBasic(params)
    }
}

impl From<Lambert> for Material {
    fn from(params: Lambert) -> Self {
        Material::Lambert(params)
    }
}

impl From<Line> for Material {
    fn from(params: Line) -> Self {
        Material::Line(params)
    }
}

impl From<Phong> for Material {
    fn from(params: Phong) -> Self {
        Material::Phong(params)
    }
}

impl From<Pbr> for Material {
    fn from(params: Pbr) -> Self {
        Material::Pbr(params)
    }
}

impl From<Sprite> for Material {
    fn from(params: Sprite) -> Self {
        Material::Sprite(params)
    }
}

impl From<Wireframe> for Material {
    fn from(params: Wireframe) -> Self {
        Material::Wireframe(params)
    }
}

impl Material {
    /// Returns true if the material is fully opaque.
    pub fn is_opaque(&self) -> bool {
        match *self {
            Material::Basic(_) => true,
            Material::CustomBasic(_) => true,
            Material::Line(_) => true,
            Material::Lambert(_) => true,
            Material::Phong(_) => true,
            Material::Pbr(_) => true,
            Material::Sprite(_) => false,
            Material::Wireframe(_) => true,
        }
    }
}
