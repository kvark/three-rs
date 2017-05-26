use std::collections::HashSet;
use std::time;
use glutin;

use {Key, Scene};
use camera::Camera;
use render::Renderer;
use factory::Factory;


struct Input {
    last_time: time::Instant,
    keys: HashSet<Key>,
    mouse_pos: (f32, f32), // normalized to NDC
}

pub struct Events {
    pub time_delta: f32,
    pub keys: HashSet<Key>,
    pub mouse_pos: (f32, f32),
}

pub struct Window<C> {
    event_loop: glutin::EventsLoop,
    input: Input,
    pub renderer: Renderer,
    pub factory: Factory,
    pub scene: Scene,
    pub camera: C,
}

impl<C: Camera> Window<C> {
    pub fn new(title: &str, camera: C) -> Self {
        let builder = glutin::WindowBuilder::new()
                             .with_title(title)
                             .with_vsync();
        let event_loop = glutin::EventsLoop::new();
        let (renderer, mut factory) = Renderer::new(builder, &event_loop);
        let scene = factory.scene();
        Window {
            event_loop: event_loop,
            input: Input {
                last_time: time::Instant::now(),
                keys: HashSet::new(),
                mouse_pos: (0.0, 0.0),
            },
            renderer: renderer,
            factory: factory,
            scene: scene,
            camera: camera,
        }
    }

    pub fn update(&mut self) -> Option<Events> {
        let mut running = true;
        let renderer = &mut self.renderer;
        let input = &mut self.input;

        self.event_loop.poll_events(|glutin::Event::WindowEvent {event, ..}| {
            use glutin::ElementState::*;
            use glutin::WindowEvent::*;
            use glutin::VirtualKeyCode as Key;
            match event {
                Resized(..) => {
                    renderer.resize();
                }
                KeyboardInput(_, _, Some(Key::Escape), _) |
                Closed => {
                    running = false
                }
                KeyboardInput(Pressed, _, Some(key), _) => {
                    input.keys.insert(key);
                }
                KeyboardInput(Released, _, Some(key), _) => {
                    input.keys.remove(&key);
                }
                MouseMoved(x, y) => {
                    input.mouse_pos = renderer.map_to_ndc(x, y);
                }
                _ => ()
            }
        });

        if running {
            let now = time::Instant::now();
            let dt = now - input.last_time;
            input.last_time = now;
            Some(Events {
                time_delta: dt.as_secs() as f32 + 1e-9 * dt.subsec_nanos() as f32,
                keys: input.keys.clone(),
                mouse_pos: input.mouse_pos,
            })
        } else {
            None
        }
    }

    pub fn render(&mut self) {
        self.camera.set_aspect(self.renderer.get_aspect());
        self.renderer.render(&self.scene, &self.camera);
    }
}
