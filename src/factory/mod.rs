mod load_gltf;

use std::{cmp, fs, io, iter, ops};
use std::borrow::Cow;
use std::collections::hash_map::{Entry, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

use animation;
use camera;
use cgmath::Vector3;
use color;
use genmesh::{Polygon, Triangulate};
use gfx;
use gfx::format::I8Norm;
use gfx::traits::{Factory as Factory_, FactoryExt};
use hub;
use image;
use itertools::Either;
use material;
use mint;
use obj;
use object;
use render;
use scene;

use audio::{AudioData, Clip, Source};
use camera::Camera;
use color::Color;
use geometry::Geometry;
use group::Group;
use hub::{Hub, HubPtr, LightData, SubLight, SubNode};
use light::{Ambient, Directional, Hemisphere, Point, ShadowMap};
use material::Material;
use mesh::{DynamicMesh, Mesh, Target, MAX_TARGETS};
use object::Object;
use render::{basic_pipe, BackendFactory, BackendResources, BasicPipelineState, DisplacementContribution, DynamicData, GpuData, ShadowFormat, Vertex, DEFAULT_VERTEX, ZEROED_DISPLACEMENT_CONTRIBUTION};
use scene::Scene;
use sprite::Sprite;
use skeleton::{Bone, Skeleton};
use text::{Font, Text, TextData};
use texture::{CubeMap, CubeMapPath, FilterMethod, Sampler, Texture, WrapMode};
use vec_map::VecMap;

const TANGENT_X: [I8Norm; 4] = [I8Norm(1), I8Norm(0), I8Norm(0), I8Norm(1)];
const NORMAL_Z: [I8Norm; 4] = [I8Norm(0), I8Norm(0), I8Norm(1), I8Norm(0)];

const QUAD: [Vertex; 4] = [
    Vertex {
        pos: [-1.0, -1.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        normal: NORMAL_Z,
        tangent: TANGENT_X,
        joint_indices: [0.0, 0.0, 0.0, 0.0],
        joint_weights: [0.0, 0.0, 0.0, 0.0],
        .. DEFAULT_VERTEX
    },
    Vertex {
        pos: [1.0, -1.0, 0.0, 1.0],
        uv: [1.0, 0.0],
        normal: NORMAL_Z,
        tangent: TANGENT_X,
        joint_indices: [0.0, 0.0, 0.0, 0.0],
        joint_weights: [0.0, 0.0, 0.0, 0.0],
        .. DEFAULT_VERTEX
    },
    Vertex {
        pos: [-1.0, 1.0, 0.0, 1.0],
        uv: [0.0, 1.0],
        normal: NORMAL_Z,
        tangent: TANGENT_X,
        joint_indices: [0.0, 0.0, 0.0, 0.0],
        joint_weights: [0.0, 0.0, 0.0, 0.0],
        .. DEFAULT_VERTEX
    },
    Vertex {
        pos: [1.0, 1.0, 0.0, 1.0],
        uv: [1.0, 1.0],
        normal: NORMAL_Z,
        tangent: TANGENT_X,
        joint_indices: [0.0, 0.0, 0.0, 0.0],
        joint_weights: [0.0, 0.0, 0.0, 0.0],
        .. DEFAULT_VERTEX
    },
];

/// Mapping writer.
pub type MapVertices<'a> = gfx::mapping::Writer<'a, BackendResources, Vertex>;

/// `Factory` is used to instantiate game objects.
pub struct Factory {
    pub(crate) backend: BackendFactory,
    hub: HubPtr,
    quad_buf: gfx::handle::Buffer<BackendResources, Vertex>,
    texture_cache: HashMap<PathBuf, Texture<[f32; 4]>>,
    default_sampler: gfx::handle::Sampler<BackendResources>,
}

/// Loaded glTF 2.0 returned by [`Factory::load_gltf`].
///
/// [`Factory::load_gltf`]: struct.Factory.html#method.load_gltf
pub struct Gltf {
    /// Imported camera views.
    pub cameras: Vec<Camera>,

    /// Imported animation clips.
    pub clips: Vec<animation::Clip>,
    
    /// The node heirarchy of the default scene.
    ///
    /// If the `glTF` contained no default scene then this
    /// container will be empty.
    pub heirarchy: VecMap<Group>,
    
    /// Imported mesh instances.
    ///
    /// ### Notes
    ///
    /// * Must be kept alive in order to be displayed.
    pub instances: Vec<Mesh>,

    /// Imported mesh materials.
    pub materials: Vec<Material>,

    /// Imported mesh templates.
    pub meshes: VecMap<Vec<Mesh>>,

    /// The root node of the default scene.
    ///
    /// If the `glTF` contained no default scene then this group
    /// will have no children.
    pub root: Group,

    /// Imported skeletons.
    pub skeletons: Vec<Skeleton>,

    /// Imported textures.
    pub textures: Vec<Texture<[f32; 4]>>,
}

impl AsRef<object::Base> for Gltf {
    fn as_ref(&self) -> &object::Base {
        self.root.as_ref()
    }
}

impl AsMut<object::Base> for Gltf {
    fn as_mut(&mut self) -> &mut object::Base {
        self.root.as_mut()
    }
}

impl Object for Gltf {}

fn f2i(x: f32) -> I8Norm {
    I8Norm(cmp::min(cmp::max((x * 127.0) as isize, -128), 127) as i8)
}

impl Factory {
    pub(crate) fn new(mut backend: BackendFactory) -> Self {
        let quad_buf = backend.create_vertex_buffer(&QUAD);
        let default_sampler = backend.create_sampler_linear();
        Factory {
            backend: backend,
            hub: Hub::new(),
            quad_buf,
            texture_cache: HashMap::new(),
            default_sampler: default_sampler,
        }
    }

    /// Create new empty [`Scene`](struct.Scene.html).
    pub fn scene(&mut self) -> Scene {
        let object = self.hub.lock().unwrap().spawn_scene();
        let hub = self.hub.clone();
        let background = scene::Background::Color(color::BLACK);
        Scene {
            object,
            hub,
            background,
        }
    }

    /// Create a new [`Bone`], one component of a [`Skeleton`].
    ///
    /// [`Bone`]: ../skeleton/struct.Bone.html
    /// [`Skeleton`]: ../skeleton/struct.Skeleton.html
    pub fn bone(&mut self) -> Bone {
        let object = self.hub.lock().unwrap().spawn_empty();
        Bone { object }
    }

    /// Create a new [`Skeleton`] from a set of [`Bone`] instances.
    ///
    /// * `bones` is the array of bones that form the skeleton.
    /// * `inverses` is an optional array of inverse bind matrices for each bone.
    /// [`Skeleton`]: ../skeleton/struct.Skeleton.html
    /// [`Bone`]: ../skeleton/struct.Bone.html
    pub fn skeleton(
        &mut self,
        bones: Vec<Bone>,
        inverse_bind_matrices: Vec<mint::ColumnMatrix4<f32>>,
    ) -> Skeleton {
        let gpu_buffer = self.backend
            .create_buffer(
                4 * bones.len(),
                gfx::buffer::Role::Constant,
                gfx::memory::Usage::Dynamic,
                gfx::SHADER_RESOURCE,
            )
            .expect("create GPU target buffer");
        let gpu_buffer_view = self.backend
            .view_buffer_as_shader_resource(&gpu_buffer)
            .expect("create shader resource view for GPU target buffer");
        let mut cpu_buffer = Vec::with_capacity(bones.len());
        for mx in &inverse_bind_matrices {
            cpu_buffer.push(mx.x.into());
            cpu_buffer.push(mx.y.into());
            cpu_buffer.push(mx.z.into());
            cpu_buffer.push(mx.w.into());
        }
        let data = hub::SkeletonData { bones, gpu_buffer, inverse_bind_matrices, gpu_buffer_view, cpu_buffer };
        let object = self.hub.lock().unwrap().spawn_skeleton(data);
        Skeleton { object }
    }

    /// Create new [Orthographic] Camera.
    /// It's used to render 2D.
    ///
    /// [Orthographic]: https://en.wikipedia.org/wiki/Orthographic_projection
    pub fn orthographic_camera<P: Into<mint::Point2<f32>>>(
        &mut self,
        center: P,
        extent_y: f32,
        range: ops::Range<f32>,
    ) -> Camera {
        Camera {
            object: self.hub.lock().unwrap().spawn_empty(),
            projection: camera::Projection::orthographic(center, extent_y, range),
        }
    }

    /// Create new [Perspective] Camera.
    ///
    /// It's used to render 3D.
    ///
    /// # Examples
    ///
    /// Creating a finite perspective camera.
    ///
    /// ```rust,no_run
    /// # #![allow(unreachable_code, unused_variables)]
    /// # let mut factory: three::Factory = unimplemented!();
    /// let camera = factory.perspective_camera(60.0, 0.1 .. 1.0);
    /// ```
    ///
    /// Creating an infinite perspective camera.
    ///
    /// ```rust,no_run
    /// # #![allow(unreachable_code, unused_variables)]
    /// # let mut factory: three::Factory = unimplemented!();
    /// let camera = factory.perspective_camera(60.0, 0.1 ..);
    /// ```
    ///
    /// [Perspective]: https://en.wikipedia.org/wiki/Perspective_(graphical)
    pub fn perspective_camera<R: Into<camera::ZRange>>(
        &mut self,
        fov_y: f32,
        range: R,
    ) -> Camera {
        Camera {
            object: self.hub.lock().unwrap().spawn_empty(),
            projection: camera::Projection::perspective(fov_y, range),
        }
    }

    /// Create empty [`Group`](struct.Group.html).
    pub fn group(&mut self) -> Group {
        Group::new(self.hub.lock().unwrap().spawn_empty())
    }

    fn mesh_vertices(geometry: &Geometry, targets: [Target; MAX_TARGETS]) -> Vec<Vertex> {
        let position_iter = geometry.vertices.iter();
        let normal_iter = if geometry.normals.is_empty() {
            Either::Left(iter::repeat(NORMAL_Z))
        } else {
            Either::Right(
                geometry
                    .normals
                    .iter()
                    .map(|n| [f2i(n.x), f2i(n.y), f2i(n.z), I8Norm(0)]),
            )
        };
        let uv_iter = if geometry.tex_coords.is_empty() {
            Either::Left(iter::repeat([0.0, 0.0]))
        } else {
            Either::Right(geometry.tex_coords.iter().map(|uv| [uv.x, uv.y]))
        };
        let tangent_iter = if geometry.tangents.is_empty() {
            // @alteous:
            // TODO: Generate tangents if texture co-ordinates are provided.
            // (Use mikktspace algorithm or otherwise.)
            Either::Left(iter::repeat(TANGENT_X))
        } else {
            Either::Right(
                geometry
                    .tangents
                    .iter()
                    .map(|t| [f2i(t.x), f2i(t.y), f2i(t.z), f2i(t.w)]),
            )
        };
        let joint_indices_iter = if geometry.joints.indices.is_empty() {
            Either::Left(iter::repeat([0.0, 0.0, 0.0, 0.0]))
        } else {
            Either::Right(geometry.joints.indices.iter().cloned())
        };
        let joint_weights_iter = if geometry.joints.weights.is_empty() {
            Either::Left(iter::repeat([1.0, 1.0, 1.0, 1.0]))
        } else {
            Either::Right(geometry.joints.weights.iter().cloned())
        };
        let (mut pi, mut ni, mut ti) = (0, 0, 0);
        let infinite_zero_vector_iter = iter::repeat(mint::Vector3::<f32> { x: 0.0, y: 0.0, z: 0.0 });
        let nr_vertices = geometry.vertices.len();
        let displacements0_iter = match targets[0] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                pi += 1;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                ni += 1;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                ti += 1;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        let displacements1_iter = match targets[1] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                pi += 1;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                ni += 1;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                ti += 1;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        let displacements2_iter = match targets[2] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                pi += 1;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                ni += 1;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                ti += 1;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        let displacements3_iter = match targets[3] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                pi += 1;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                ni += 1;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                ti += 1;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        let displacements4_iter = match targets[4] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                pi += 1;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                ni += 1;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                ti += 1;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        let displacements5_iter = match targets[5] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                pi += 1;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                ni += 1;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                ti += 1;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        let displacements6_iter = match targets[6] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                pi += 1;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                ni += 1;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                ti += 1;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        let displacements7_iter = match targets[7] {
            Target::Position => {
                let begin = pi * nr_vertices;
                let end = begin + nr_vertices;
                Either::Left(geometry.morph_targets.vertices[begin .. end].iter().cloned())
            }
            Target::Normal => {
                let begin = ni * nr_vertices;
                let end = begin + nr_vertices;
                Either::Left(geometry.morph_targets.normals[begin .. end].iter().cloned())
            }
            Target::Tangent => {
                let begin = ti * nr_vertices;
                let end = begin + nr_vertices;
                Either::Left(geometry.morph_targets.tangents[begin .. end].iter().cloned())
            }
            Target::None => {
                Either::Right(infinite_zero_vector_iter.clone())
            }
        };
        izip!(
            position_iter,
            normal_iter,
            tangent_iter,
            uv_iter,
            joint_indices_iter,
            joint_weights_iter,
            izip!(
                displacements0_iter,
                displacements1_iter,
                displacements2_iter,
                displacements3_iter,
                displacements4_iter,
                displacements5_iter,
                displacements6_iter,
                displacements7_iter
            )
        )
            .map(|(pos, normal, tangent, uv, joint_indices, joint_weights, (d0, d1, d2, d3, d4, d5, d6, d7))| {
                Vertex {
                    pos: [pos.x, pos.y, pos.z, 1.0],
                    normal,
                    uv,
                    tangent,
                    joint_indices,
                    joint_weights,
                    displacement0: [d0.x, d0.y, d0.z, 0.0],
                    displacement1: [d1.x, d1.y, d1.z, 0.0],
                    displacement2: [d2.x, d2.y, d2.z, 0.0],
                    displacement3: [d3.x, d3.y, d3.z, 0.0],
                    displacement4: [d4.x, d4.y, d4.z, 0.0],
                    displacement5: [d5.x, d5.y, d5.z, 0.0],
                    displacement6: [d6.x, d6.y, d6.z, 0.0],
                    displacement7: [d7.x, d7.y, d7.z, 0.0],
                }
            })
            .collect()
    }

    /// Create new `Mesh` with desired `Geometry` and `Material`.
    pub fn mesh<M: Into<Material>>(
        &mut self,
        geometry: Geometry,
        material: M,
    ) -> Mesh {
        self.mesh_with_targets(geometry, material, [Target::None; MAX_TARGETS])
    }

    /// Create new `Mesh` mesh with desired `Geometry`, `Material`, and
    /// morph `Target` bindings.
    pub fn mesh_with_targets<M: Into<Material>>(
        &mut self,
        geometry: Geometry,
        material: M,
        targets: [Target; MAX_TARGETS],
    ) -> Mesh {
        let vertices = Self::mesh_vertices(&geometry, targets);
        let cbuf = self.backend.create_constant_buffer(1);
        let (vbuf, slice) = if geometry.faces.is_empty() {
            self.backend.create_vertex_buffer_with_slice(&vertices, ())
        } else {
            let faces: &[u32] = gfx::memory::cast_slice(&geometry.faces);
            self.backend
                .create_vertex_buffer_with_slice(&vertices, faces)
        };
        let mut dcs = [DisplacementContribution::default(); MAX_TARGETS];
        for i in 0 .. MAX_TARGETS {
            match targets[i] {
                Target::Position => dcs[i].position = 1.0,
                Target::Normal => dcs[i].normal = 1.0,
                Target::Tangent => dcs[i].tangent = 1.0,
                Target::None => {},
            }
        }
        Mesh {
            object: self.hub.lock().unwrap().spawn_visual(
                material.into(),
                GpuData {
                    slice,
                    vertices: vbuf,
                    constants: cbuf,
                    pending: None,
                    displacement_contributions: dcs,
                },
                None,
            ),
        }
    }

    /// Create a new `DynamicMesh` with desired `Geometry` and `Material`.
    pub fn mesh_dynamic<M: Into<Material>>(
        &mut self,
        geometry: Geometry,
        material: M,
    ) -> DynamicMesh {
        let slice = {
            let data: &[u32] = gfx::memory::cast_slice(&geometry.faces);
            gfx::Slice {
                start: 0,
                end: data.len() as u32,
                base_vertex: 0,
                instances: None,
                buffer: self.backend.create_index_buffer(data),
            }
        };
        let (num_vertices, vertices, upload_buf) = {
            let data = Self::mesh_vertices(&geometry, [Target::None; MAX_TARGETS]);
            let dest_buf = self.backend
                .create_buffer_immutable(&data, gfx::buffer::Role::Vertex, gfx::memory::TRANSFER_DST)
                .unwrap();
            let upload_buf = self.backend.create_upload_buffer(data.len()).unwrap();
            // TODO: Workaround for not having a 'write-to-slice' capability.
            // Reason: The renderer copies the entire staging buffer upon updates.
            {
                self.backend
                    .write_mapping(&upload_buf)
                    .unwrap()
                    .copy_from_slice(&data);
            }
            (data.len(), dest_buf, upload_buf)
        };
        let constants = self.backend.create_constant_buffer(1);

        DynamicMesh {
            object: self.hub.lock().unwrap().spawn_visual(
                material.into(),
                GpuData {
                    slice,
                    vertices,
                    constants,
                    pending: None,
                    displacement_contributions: ZEROED_DISPLACEMENT_CONTRIBUTION,
                },
                None,
            ),
            geometry,
            dynamic: DynamicData {
                num_vertices,
                buffer: upload_buf,
            },
        }
    }

    /// Create a `Mesh` sharing the geometry with another one.
    /// Rendering a sequence of meshes with the same geometry is faster.
    /// The material is duplicated from the template.
    pub fn mesh_instance(
        &mut self,
        template: &Mesh,
    ) -> Mesh {
        let mut hub = self.hub.lock().unwrap();
        let gpu_data = match hub.get(&template).sub_node {
            SubNode::Visual(_, ref gpu, _) => GpuData {
                constants: self.backend.create_constant_buffer(1),
                ..gpu.clone()
            },
            _ => unreachable!(),
        };
        let material = match hub.get(&template).sub_node {
            SubNode::Visual(ref mat, _, _) => mat.clone(),
            _ => unreachable!(),
        };
        Mesh {
            object: hub.spawn_visual(material, gpu_data, None),
        }
    }

    /// Create a `Mesh` sharing the geometry with another one but with a different material.
    /// Rendering a sequence of meshes with the same geometry is faster.
    pub fn mesh_instance_with_material<M: Into<Material>>(
        &mut self,
        template: &Mesh,
        material: M,
    ) -> Mesh {
        let mut hub = self.hub.lock().unwrap();
        let gpu_data = match hub.get(&template).sub_node {
            SubNode::Visual(_, ref gpu, _) => GpuData {
                constants: self.backend.create_constant_buffer(1),
                ..gpu.clone()
            },
            _ => unreachable!(),
        };
        Mesh {
            object: hub.spawn_visual(material.into(), gpu_data, None),
        }
    }

    /// Create new sprite from `Material`.
    pub fn sprite(
        &mut self,
        material: material::Sprite,
    ) -> Sprite {
        Sprite::new(self.hub.lock().unwrap().spawn_visual(
            material.into(),
            GpuData {
                slice: gfx::Slice::new_match_vertex_buffer(&self.quad_buf),
                vertices: self.quad_buf.clone(),
                constants: self.backend.create_constant_buffer(1),
                pending: None,
                displacement_contributions: ZEROED_DISPLACEMENT_CONTRIBUTION,
            },
            None,
        ))
    }

    /// Create new `AmbientLight`.
    pub fn ambient_light(
        &mut self,
        color: Color,
        intensity: f32,
    ) -> Ambient {
        Ambient::new(self.hub.lock().unwrap().spawn_light(LightData {
            color,
            intensity,
            sub_light: SubLight::Ambient,
            shadow: None,
        }))
    }

    /// Create new `DirectionalLight`.
    pub fn directional_light(
        &mut self,
        color: Color,
        intensity: f32,
    ) -> Directional {
        Directional::new(self.hub.lock().unwrap().spawn_light(LightData {
            color,
            intensity,
            sub_light: SubLight::Directional,
            shadow: None,
        }))
    }

    /// Create new `HemisphereLight`.
    pub fn hemisphere_light(
        &mut self,
        sky_color: Color,
        ground_color: Color,
        intensity: f32,
    ) -> Hemisphere {
        Hemisphere::new(self.hub.lock().unwrap().spawn_light(LightData {
            color: sky_color,
            intensity,
            sub_light: SubLight::Hemisphere {
                ground: ground_color,
            },
            shadow: None,
        }))
    }

    /// Create new `PointLight`.
    pub fn point_light(
        &mut self,
        color: Color,
        intensity: f32,
    ) -> Point {
        Point::new(self.hub.lock().unwrap().spawn_light(LightData {
            color,
            intensity,
            sub_light: SubLight::Point,
            shadow: None,
        }))
    }

    /// Create a `Sampler` with default properties.
    ///
    /// The default sampler has `Clamp` as its horizontal and vertical
    /// wrapping mode and `Scale` as its filtering method.
    pub fn default_sampler(&self) -> Sampler {
        Sampler(self.default_sampler.clone())
    }

    /// Create new `Sampler`.
    pub fn sampler(
        &mut self,
        filter_method: FilterMethod,
        horizontal_wrap_mode: WrapMode,
        vertical_wrap_mode: WrapMode,
    ) -> Sampler {
        use gfx::texture::Lod;
        let info = gfx::texture::SamplerInfo {
            filter: filter_method,
            wrap_mode: (horizontal_wrap_mode, vertical_wrap_mode, WrapMode::Clamp),
            lod_bias: Lod::from(0.0),
            lod_range: (Lod::from(-8000.0), Lod::from(8000.0)),
            comparison: None,
            border: gfx::texture::PackedColor(0),
        };
        let inner = self.backend.create_sampler(info);
        Sampler(inner)
    }

    /// Create new `ShadowMap`.
    pub fn shadow_map(
        &mut self,
        width: u16,
        height: u16,
    ) -> ShadowMap {
        let (_, resource, target) = self.backend
            .create_depth_stencil::<ShadowFormat>(width, height)
            .unwrap();
        ShadowMap { resource, target }
    }

    /// Create a basic mesh pipeline using a custom shader.
    pub fn basic_pipeline<P: AsRef<Path>>(
        &mut self,
        dir: P,
        name: &str,
        primitive: gfx::Primitive,
        rasterizer: gfx::state::Rasterizer,
        color_mask: gfx::state::ColorMask,
        blend_state: gfx::state::Blend,
        depth_state: gfx::state::Depth,
        stencil_state: gfx::state::Stencil,
    ) -> Result<BasicPipelineState, render::PipelineCreationError> {
        use gfx::traits::FactoryExt;
        let vs = render::Source::user(&dir, name, "vs")?;
        let ps = render::Source::user(&dir, name, "ps")?;
        let shaders = self.backend
            .create_shader_set(vs.0.as_bytes(), ps.0.as_bytes())?;
        let init = basic_pipe::Init {
            out_color: ("Target0", color_mask, blend_state),
            out_depth: (depth_state, stencil_state),
            ..basic_pipe::new()
        };
        let pso = self.backend
            .create_pipeline_state(&shaders, primitive, rasterizer, init)?;
        Ok(pso)
    }
    
    /// Create new UI (on-screen) text. See [`Text`](struct.Text.html) for default settings.
    pub fn ui_text<S: Into<String>>(
        &mut self,
        font: &Font,
        text: S,
    ) -> Text {
        let data = TextData::new(font, text);
        let object = self.hub.lock().unwrap().spawn_ui_text(data);
        Text::with_object(object)
    }

    /// Create new audio source.
    pub fn audio_source(&mut self) -> Source {
        let data = AudioData::new();
        let object = self.hub.lock().unwrap().spawn_audio_source(data);
        Source::with_object(object)
    }

    /// Map vertices for updating their data.
    pub fn map_vertices<'a>(
        &'a mut self,
        mesh: &'a mut DynamicMesh,
    ) -> MapVertices<'a> {
        self.hub.lock().unwrap().update_mesh(mesh);
        self.backend.write_mapping(&mesh.dynamic.buffer).unwrap()
    }

    /// Interpolate between the shapes of a `DynamicMesh`.
    pub fn mix(
        &mut self,
        mesh: &DynamicMesh,
        shapes: &[(&str, f32)],
    ) {
        self.hub.lock().unwrap().update_mesh(mesh);
        let targets: Vec<(usize, f32)> = shapes
            .iter()
            .filter_map(|&(name, k)| {
                mesh.geometry.morph_targets.names
                    .iter()
                    .find(|&(_, entry)| entry == name)
                    .map(|(idx, _)| (idx, k))
            })
            .collect();
        let mut mapping = self.backend.write_mapping(&mesh.dynamic.buffer).unwrap();

        let n = mesh.geometry.vertices.len();
        for i in 0 .. n {
            let (mut pos, ksum) = targets.iter().fold(
                (Vector3::new(0.0, 0.0, 0.0), 0.0),
                |(pos, ksum), &(idx, k)| {
                    let p: [f32; 3] = mesh.geometry.morph_targets.vertices[idx * n + i].into();
                    (pos + k * Vector3::from(p), ksum + k)
                },
            );
            if ksum != 1.0 {
                let p: [f32; 3] = mesh.geometry.vertices[i].into();
                pos += (1.0 - ksum) * Vector3::from(p);
            }
            mapping[i] = Vertex {
                pos: [pos.x, pos.y, pos.z, 1.0],
                .. mapping[i]
            };
        }
    }

    /// Load TrueTypeFont (.ttf) from file.
    /// #### Panics
    /// Panics if I/O operations with file fails (e.g. file not found or corrupted)
    pub fn load_font<P: AsRef<Path>>(
        &mut self,
        file_path: P,
    ) -> Font {
        use self::io::Read;
        let file_path = file_path.as_ref();
        let mut buffer = Vec::new();
        let file = fs::File::open(&file_path).expect(&format!(
            "Can't open font file:\nFile: {}",
            file_path.display()
        ));
        io::BufReader::new(file)
            .read_to_end(&mut buffer)
            .expect(&format!(
                "Can't read font file:\nFile: {}",
                file_path.display()
            ));
        Font::new(buffer, file_path.to_owned(), self.backend.clone())
    }

    fn parse_texture_format(path: &Path) -> image::ImageFormat {
        use image::ImageFormat as F;
        let extension = path.extension()
            .expect("no extension for an image?")
            .to_string_lossy()
            .to_lowercase();
        match extension.as_str() {
            "png" => F::PNG,
            "jpg" | "jpeg" => F::JPEG,
            "gif" => F::GIF,
            "webp" => F::WEBP,
            "ppm" => F::PPM,
            "tiff" => F::TIFF,
            "tga" => F::TGA,
            "bmp" => F::BMP,
            "ico" => F::ICO,
            "hdr" => F::HDR,
            _ => panic!("Unrecognized image extension: {}", extension),
        }
    }

    fn load_texture_impl(
        path: &Path,
        sampler: Sampler,
        factory: &mut BackendFactory,
    ) -> Texture<[f32; 4]> {
        use gfx::texture as t;
        let format = Factory::parse_texture_format(path);
        let file = fs::File::open(path).unwrap_or_else(|e| panic!("Unable to open {}: {:?}", path.display(), e));
        let img = image::load(io::BufReader::new(file), format)
            .unwrap_or_else(|e| panic!("Unable to decode {}: {:?}", path.display(), e))
            .flipv()
            .to_rgba();
        let (width, height) = img.dimensions();
        let kind = t::Kind::D2(width as t::Size, height as t::Size, t::AaMode::Single);
        let mipmap = gfx::texture::Mipmap::Allocated;
        let (_, view) = factory
            .create_texture_immutable_u8::<gfx::format::Srgba8>(kind, mipmap, &[&img])
            .unwrap_or_else(|e| {
                panic!(
                    "Unable to create GPU texture for {}: {:?}",
                    path.display(),
                    e
                )
            });
        Texture::new(view, sampler.0, [width, height])
    }

    fn load_cubemap_impl<P: AsRef<Path>>(
        paths: &CubeMapPath<P>,
        sampler: Sampler,
        factory: &mut BackendFactory,
    ) -> CubeMap<[f32; 4]> {
        use gfx::texture as t;
        let images = paths
            .as_array()
            .iter()
            .map(|path| {
                let format = Factory::parse_texture_format(path.as_ref());
                let file = fs::File::open(path).unwrap_or_else(|e| {
                    panic!("Unable to open {}: {:?}", path.as_ref().display(), e)
                });
                image::load(io::BufReader::new(file), format)
                    .unwrap_or_else(|e| {
                        panic!("Unable to decode {}: {:?}", path.as_ref().display(), e)
                    })
                    .to_rgba()
            })
            .collect::<Vec<_>>();
        let data: [&[u8]; 6] = [
            &images[0],
            &images[1],
            &images[2],
            &images[3],
            &images[4],
            &images[5],
        ];
        let size = images[0].dimensions().0;
        let kind = t::Kind::Cube(size as t::Size);
        let (_, view) = factory
            .create_texture_immutable_u8::<gfx::format::Srgba8>(kind, gfx::texture::Mipmap::Allocated, &data)
            .unwrap_or_else(|e| {
                panic!("Unable to create GPU texture for cubemap: {:?}", e);
            });
        CubeMap::new(view, sampler.0)
    }

    fn request_texture<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Texture<[f32; 4]> {
        let sampler = self.default_sampler();
        match self.texture_cache.entry(path.as_ref().to_owned()) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => {
                let tex = Self::load_texture_impl(path.as_ref(), sampler, &mut self.backend);
                e.insert(tex.clone());
                tex
            }
        }
    }

    fn load_obj_material(
        &mut self,
        mat: &obj::Material,
        has_normals: bool,
        has_uv: bool,
        obj_dir: Option<&Path>,
    ) -> Material {
        let cf2u = |c: [f32; 3]| {
            c.iter()
                .fold(0, |u, &v| (u << 8) + cmp::min((v * 255.0) as u32, 0xFF))
        };
        match *mat {
            obj::Material {
                kd: Some(color),
                ns: Some(glossiness),
                ..
            } if has_normals =>
            {
                material::Phong {
                    color: cf2u(color),
                    glossiness,
                }.into()
            }
            obj::Material {
                kd: Some(color), ..
            } if has_normals =>
            {
                material::Lambert {
                    color: cf2u(color),
                    flat: false,
                }.into()
            }
            obj::Material {
                kd: Some(color),
                ref map_kd,
                ..
            } => material::Basic {
                color: cf2u(color),
                map: match (has_uv, map_kd) {
                    (true, &Some(ref name)) => Some(self.request_texture(&concat_path(obj_dir, name))),
                    _ => None,
                },
            }.into(),
            _ => material::Basic {
                color: 0xffffff,
                map: None,
            }.into(),
        }
    }

    /// Load texture from pre-loaded data.
    pub fn load_texture_from_memory(
        &mut self,
        width: u16,
        height: u16,
        pixels: &[u8],
        sampler: Sampler,
    ) -> Texture<[f32; 4]> {
        use gfx::texture as t;
        let kind = t::Kind::D2(width, height, t::AaMode::Single);
        let (_, view) = self.backend
            .create_texture_immutable_u8::<gfx::format::Srgba8>(kind, gfx::texture::Mipmap::Allocated, &[pixels])
            .unwrap_or_else(|e| {
                panic!("Unable to create GPU texture from memory: {:?}", e);
            });
        Texture::new(view, sampler.0, [width as u32, height as u32])
    }

    /// Load texture from file.
    /// Supported file formats are: PNG, JPEG, GIF, WEBP, PPM, TIFF, TGA, BMP, ICO, HDR.
    pub fn load_texture<P: AsRef<Path>>(
        &mut self,
        path_str: P,
    ) -> Texture<[f32; 4]> {
        self.request_texture(path_str)
    }

    /// Load cubemap from files.
    /// Supported file formats are: PNG, JPEG, GIF, WEBP, PPM, TIFF, TGA, BMP, ICO, HDR.
    pub fn load_cubemap<P: AsRef<Path>>(
        &mut self,
        paths: &CubeMapPath<P>,
    ) -> CubeMap<[f32; 4]> {
        Factory::load_cubemap_impl(paths, self.default_sampler(), &mut self.backend)
    }

    /// Load mesh from Wavefront Obj format.
    /// #### Note
    /// You must store `Vec<Mesh>` somewhere to keep them alive.
    pub fn load_obj(
        &mut self,
        path_str: &str,
    ) -> (HashMap<String, Group>, Vec<Mesh>) {
        use genmesh::{Indexer, LruIndexer, Vertices};

        info!("Loading {}", path_str);
        let path = Path::new(path_str);
        let path_parent = path.parent();
        let obj = obj::load::<Polygon<obj::IndexTuple>>(path).unwrap();

        let hub_ptr = self.hub.clone();
        let mut hub = hub_ptr.lock().unwrap();
        let mut groups = HashMap::new();
        let mut meshes = Vec::new();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for object in obj.object_iter() {
            let group = Group::new(hub.spawn_empty());
            for gr in object.group_iter() {
                let (mut num_normals, mut num_uvs) = (0, 0);
                {
                    // separate scope for LruIndexer
                    let f2i = |x: f32| I8Norm(cmp::min(cmp::max((x * 127.) as isize, -128), 127) as i8);
                    vertices.clear();
                    let mut lru = LruIndexer::new(10, |_, (ipos, iuv, inor)| {
                        let p: [f32; 3] = obj.position()[ipos];
                        vertices.push(Vertex {
                            pos: [p[0], p[1], p[2], 1.0],
                            uv: match iuv {
                                Some(i) => {
                                    num_uvs += 1;
                                    obj.texture()[i]
                                }
                                None => [0.0, 0.0],
                            },
                            normal: match inor {
                                Some(id) => {
                                    num_normals += 1;
                                    let n: [f32; 3] = obj.normal()[id];
                                    [f2i(n[0]), f2i(n[1]), f2i(n[2]), I8Norm(0)]
                                }
                                None => [I8Norm(0), I8Norm(0), I8Norm(0x7f), I8Norm(0)],
                            },
                            tangent: TANGENT_X, // TODO
                            joint_indices: [0.0; 4],
                            joint_weights: [0.0; 4],
                            .. DEFAULT_VERTEX
                        });
                    });

                    indices.clear();
                    indices.extend(
                        gr.indices
                            .iter()
                            .cloned()
                            .triangulate()
                            .vertices()
                            .map(|tuple| lru.index(tuple) as u16),
                    );
                };

                info!(
                    "\tmaterial {} with {} normals and {} uvs",
                    gr.name,
                    num_normals,
                    num_uvs
                );
                let material = match gr.material {
                    Some(ref rc_mat) => self.load_obj_material(&*rc_mat, num_normals != 0, num_uvs != 0, path_parent),
                    None => material::Basic {
                        color: 0xFFFFFF,
                        map: None,
                    }.into(),
                };
                info!("\t{:?}", material);

                let (vbuf, slice) = self.backend
                    .create_vertex_buffer_with_slice(&vertices, &indices[..]);
                let cbuf = self.backend.create_constant_buffer(1);
                let mut mesh = Mesh {
                    object: hub.spawn_visual(
                        material,
                        GpuData {
                            slice,
                            vertices: vbuf,
                            constants: cbuf,
                            pending: None,
                            displacement_contributions: ZEROED_DISPLACEMENT_CONTRIBUTION,
                        },
                        None,
                    ),
                };
                mesh.set_parent(&group);
                meshes.push(mesh);
            }

            groups.insert(object.name.clone(), group);
        }

        (groups, meshes)
    }

    /// Load audio from file. Supported formats are Flac, Vorbis and WAV.
    pub fn load_audio<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Clip {
        let mut buffer = Vec::new();
        let mut file = fs::File::open(&path).expect(&format!(
            "Can't open audio file:\nFile: {}",
            path.as_ref().display()
        ));
        file.read_to_end(&mut buffer).expect(&format!(
            "Can't read audio file:\nFile: {}",
            path.as_ref().display()
        ));
        Clip::new(buffer)
    }
}

fn concat_path<'a>(
    base: Option<&Path>,
    name: &'a str,
) -> Cow<'a, Path> {
    match base {
        Some(base) => Cow::Owned(base.join(name)),
        None => Cow::Borrowed(Path::new(name)),
    }
}
