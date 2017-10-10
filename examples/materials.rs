extern crate three;

fn main() {
    let mut win = three::Window::new("Three-rs materials example");
    let mut cam = win.factory.perspective_camera(75.0, 1.0 .. 50.0);
    cam.set_position([0.0, 0.0, 10.0]);

    let mut light = win.factory.point_light(0xffffff, 0.5);
    let mut pos = [0.0, 5.0, 5.0];
    light.set_position(pos);
    win.scene.add(&light);

    let geometry = three::Geometry::cylinder(1.0, 2.0, 2.0, 5);
    let mut materials: Vec<three::Material> = vec![
        three::material::Basic {
            color: 0xFFFFFF,
            map: None,
            ..Default::default()
        }.into(),
        three::material::Lambert {
            color: 0xFFFFFF,
            flat: true,
        }.into(),
        three::material::Lambert {
            color: 0xFFFFFF,
            flat: false,
        }.into(),
        three::material::Phong {
            color: 0xFFFFFF,
            glossiness: 80.0,
        }.into(),
        three::material::Pbr {
            base_color_factor: 0xFFFFFF,
            base_color_alpha: 1.0,
            metallic_factor: 0.5,
            roughness_factor: 0.5,
            occlusion_strength: 0.2,
            emissive_factor: 0x000000,
            normal_scale: 1.0,
            base_color_map: None,
            normal_map: None,
            emissive_map: None,
            metallic_roughness_map: None,
            occlusion_map: None,
        }.into(),
    ];
    let count = materials.len();

    let _cubes: Vec<_> = materials
        .drain(..)
        .enumerate()
        .map(|(i, mat)| {
            let offset = 4.0 * (i as f32 + 0.5 - 0.5 * count as f32);
            let mut mesh = win.factory.mesh(geometry.clone(), mat);
            mesh.set_position([offset, 0.0, 0.0]);
            win.scene.add(&mesh);
            mesh
        })
        .collect();

    while win.update() && !three::KEY_ESCAPE.is_hit(&win.input) {
        if let Some(diff) = three::AXIS_LEFT_RIGHT.timed(&win.input) {
            pos[0] += 5.0 * diff;
            light.set_position(pos);
        }

        win.render(&cam);
    }
}
