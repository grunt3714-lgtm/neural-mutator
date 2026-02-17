use pyo3::prelude::*;
use rand::Rng;
use std::sync::mpsc;

const NUM_FOODS: usize = 10;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuState {
    grid_size: u32,
    max_steps: u32,
    step_count: u32,
    direction: i32,
    head_r: i32,
    head_c: i32,
    num_foods: u32,
    done: u32,
    reward: f32,
    terminated: u32,
    truncated: u32,
    length: u32,
    action: i32,
    rng_state: u32,
    food_r: [i32; NUM_FOODS],
    food_c: [i32; NUM_FOODS],
    _pad: [u32; 2],  // pad to 96 bytes (aligned)
}

struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    state_buffer: wgpu::Buffer,
    body_buffer: wgpu::Buffer,
    pixels_buffer: wgpu::Buffer,
    state_readback: wgpu::Buffer,
    pixels_readback: wgpu::Buffer,
    body_readback: wgpu::Buffer,
    pixel_count: usize,
    max_cells: usize,
}

impl GpuRuntime {
    fn new(grid_size: usize) -> Result<(Self, String, String), String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })
        .ok_or_else(|| "No wgpu adapter available".to_string())?;

        let info = adapter.get_info();
        let backend = format!("{:?}", info.backend);
        let adapter_name = info.name;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("snake_device"),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("request_device failed: {e}"))?;

        let max_cells = grid_size * grid_size;
        let pixel_count = max_cells;

        let shader_src = format!(
            r#"
const NUM_FOODS: u32 = {num_foods}u;

struct State {{
  grid_size: u32,
  max_steps: u32,
  step_count: u32,
  direction: i32,
  head_r: i32,
  head_c: i32,
  num_foods: u32,
  done: u32,
  reward: f32,
  terminated: u32,
  truncated: u32,
  length: u32,
  action: i32,
  rng_state: u32,
  food_r: array<i32, {num_foods}>,
  food_c: array<i32, {num_foods}>,
  pad0: u32,
  pad1: u32,
}};

@group(0) @binding(0) var<storage, read_write> state: State;
@group(0) @binding(1) var<storage, read_write> body: array<vec2<i32>, {max_cells}>;
@group(0) @binding(2) var<storage, read_write> pixels: array<u32, {pixel_count}>;

fn rgb(r: u32, g: u32, b: u32) -> u32 {{
  return r | (g << 8u) | (b << 16u);
}}

fn pcg_next() -> u32 {{
  var s = state.rng_state;
  s = s * 747796405u + 2891336453u;
  state.rng_state = s;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}}

fn is_occupied(rr: i32, cc: i32, len: u32) -> bool {{
  for (var i: u32 = 0u; i < len; i = i + 1u) {{
    if (body[i].x == rr && body[i].y == cc) {{
      return true;
    }}
  }}
  return false;
}}

fn is_food(rr: i32, cc: i32) -> bool {{
  for (var i: u32 = 0u; i < NUM_FOODS; i = i + 1u) {{
    if (state.food_r[i] == rr && state.food_c[i] == cc) {{
      return true;
    }}
  }}
  return false;
}}

fn spawn_one_food(slot: u32) {{
  let g = state.grid_size;
  let cells = g * g;
  let min_dist: i32 = 4;
  var far_count: u32 = 0u;
  var any_count: u32 = 0u;
  for (var ci: u32 = 0u; ci < cells; ci = ci + 1u) {{
    let rr = i32(ci / g);
    let cc = i32(ci % g);
    if (!is_occupied(rr, cc, state.length) && !is_food(rr, cc)) {{
      any_count = any_count + 1u;
      let d = abs(rr - state.head_r) + abs(cc - state.head_c);
      if (d >= min_dist) {{
        far_count = far_count + 1u;
      }}
    }}
  }}
  var use_far = far_count > 0u;
  var target_count = far_count;
  if (!use_far) {{
    target_count = any_count;
  }}
  if (target_count > 0u) {{
    let pick = pcg_next() % target_count;
    var idx: u32 = 0u;
    for (var ci: u32 = 0u; ci < cells; ci = ci + 1u) {{
      let rr = i32(ci / g);
      let cc = i32(ci % g);
      if (!is_occupied(rr, cc, state.length) && !is_food(rr, cc)) {{
        var valid = true;
        if (use_far) {{
          let d = abs(rr - state.head_r) + abs(cc - state.head_c);
          if (d < min_dist) {{
            valid = false;
          }}
        }}
        if (valid) {{
          if (idx == pick) {{
            state.food_r[slot] = rr;
            state.food_c[slot] = cc;
            return;
          }}
          idx = idx + 1u;
        }}
      }}
    }}
  }}
}}

fn render() {{
  let g = state.grid_size;
  let cells = g * g;
  for (var i: u32 = 0u; i < cells; i = i + 1u) {{
    pixels[i] = 0u;
  }}

  // Render all foods
  for (var f: u32 = 0u; f < NUM_FOODS; f = f + 1u) {{
    let fr = state.food_r[f];
    let fc = state.food_c[f];
    if (fr >= 0 && fc >= 0) {{
      let fi = u32(fr) * g + u32(fc);
      pixels[fi] = rgb(255u, 0u, 0u);
    }}
  }}

  for (var i: u32 = 1u; i < state.length; i = i + 1u) {{
    let r = body[i].x;
    let c = body[i].y;
    if (r >= 0 && c >= 0) {{
      let bi = u32(r) * g + u32(c);
      pixels[bi] = rgb(0u, 255u, 0u);
    }}
  }}

  if (state.head_r >= 0 && state.head_c >= 0) {{
    let hi = u32(state.head_r) * g + u32(state.head_c);
    pixels[hi] = rgb(255u, 255u, 255u);
  }}
}}

@compute @workgroup_size(1)
fn main() {{
  state.reward = 0.0;
  state.terminated = 0u;
  state.truncated = 0u;

  if (state.action < 0) {{
    render();
    return;
  }}

  if (state.done != 0u) {{
    state.reward = 0.0;
    state.terminated = 1u;
    render();
    return;
  }}

  let opposite = (state.direction + 2) % 4;
  var a = state.action;
  if (a == opposite) {{
    a = state.direction;
  }}
  state.direction = a;

  var dr = 0;
  var dc = 0;
  if (a == 0) {{
    dr = -1;
    dc = 0;
  }} else if (a == 1) {{
    dr = 0;
    dc = 1;
  }} else if (a == 2) {{
    dr = 1;
    dc = 0;
  }} else {{
    dr = 0;
    dc = -1;
  }}

  let nr = state.head_r + dr;
  let nc = state.head_c + dc;

  state.step_count = state.step_count + 1u;
  state.reward = -0.01;

  let g = i32(state.grid_size);
  if (nr < 0 || nr >= g || nc < 0 || nc >= g) {{
    state.done = 1u;
    state.reward = -1.0;
    state.terminated = 1u;
    render();
    return;
  }}

  // Check which food (if any) was eaten
  var ate_food: i32 = -1;
  for (var f: u32 = 0u; f < NUM_FOODS; f = f + 1u) {{
    if (nr == state.food_r[f] && nc == state.food_c[f]) {{
      ate_food = i32(f);
      break;
    }}
  }}
  let ate = ate_food >= 0;
  let tail = body[state.length - 1u];

  var hits_body = false;
  for (var i: u32 = 0u; i < state.length; i = i + 1u) {{
    if (body[i].x == nr && body[i].y == nc) {{
      hits_body = true;
      break;
    }}
  }}

  if (hits_body && !((nr == tail.x && nc == tail.y) && !ate)) {{
    state.done = 1u;
    state.reward = -1.0;
    state.terminated = 1u;
    render();
    return;
  }}

  var new_len = state.length;
  if (ate) {{
    new_len = new_len + 1u;
  }}

  var i = new_len - 1u;
  loop {{
    if (i == 0u) {{
      break;
    }}
    body[i] = body[i - 1u];
    i = i - 1u;
  }}
  body[0] = vec2<i32>(nr, nc);
  state.head_r = nr;
  state.head_c = nc;
  state.length = new_len;

  if (ate) {{
    state.reward = 1.0;
    // Respawn the eaten food slot
    spawn_one_food(u32(ate_food));
  }}

  if (state.step_count >= state.max_steps) {{
    state.done = 1u;
    state.truncated = 1u;
  }}

  render();
}}
            "#,
            num_foods = NUM_FOODS,
            max_cells = max_cells,
            pixel_count = pixel_count,
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("snake_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_buffer"),
            size: std::mem::size_of::<GpuState>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let body_bytes = (max_cells * std::mem::size_of::<[i32; 2]>()) as u64;
        let body_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body_buffer"),
            size: body_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pixel_bytes = (pixel_count * std::mem::size_of::<u32>()) as u64;
        let pixels_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels_buffer"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let state_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_readback"),
            size: std::mem::size_of::<GpuState>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pixels_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels_readback"),
            size: pixel_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let body_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body_readback"),
            size: body_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("snake_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("snake_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("snake_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snake_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: body_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pixels_buffer.as_entire_binding(),
                },
            ],
        });

        Ok((
            Self {
                device,
                queue,
                pipeline,
                bind_group,
                state_buffer,
                body_buffer,
                pixels_buffer,
                state_readback,
                pixels_readback,
                body_readback,
                pixel_count,
                max_cells,
            },
            backend,
            adapter_name,
        ))
    }

    fn write_state_and_body(&self, state: &GpuState, body: &[[i32; 2]]) {
        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::bytes_of(state));
        self.queue
            .write_buffer(&self.body_buffer, 0, bytemuck::cast_slice(body));
    }

    fn run(&self) -> Result<(GpuState, Vec<u32>, Vec<[i32; 2]>), String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("snake_encoder") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("snake_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.state_buffer,
            0,
            &self.state_readback,
            0,
            std::mem::size_of::<GpuState>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.pixels_buffer,
            0,
            &self.pixels_readback,
            0,
            (self.pixel_count * std::mem::size_of::<u32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.body_buffer,
            0,
            &self.body_readback,
            0,
            (self.max_cells * std::mem::size_of::<[i32; 2]>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let state_bytes = Self::map_read(&self.device, &self.state_readback)?;
        let pixel_bytes = Self::map_read(&self.device, &self.pixels_readback)?;
        let body_bytes = Self::map_read(&self.device, &self.body_readback)?;

        let state: GpuState = *bytemuck::from_bytes(&state_bytes[..std::mem::size_of::<GpuState>()]);
        let pixels: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&pixel_bytes).to_vec();
        let body: Vec<[i32; 2]> = bytemuck::cast_slice::<u8, [i32; 2]>(&body_bytes).to_vec();

        Ok((state, pixels, body))
    }

    fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, String> {
        let slice = buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("map_async recv failed: {e}"))?
            .map_err(|e| format!("map_async failed: {e}"))?;

        let data = slice.get_mapped_range().to_vec();
        buffer.unmap();
        Ok(data)
    }
}

#[pyclass]
pub struct SnakeGpuCore {
    grid_size: usize,
    max_steps: usize,
    num_foods: usize,
    step_count: usize,
    direction: i32,
    head_r: i32,
    head_c: i32,
    body: Vec<(i32, i32)>,
    foods: Vec<(i32, i32)>,
    done: bool,
    rng_state: u32,
    backend: String,
    adapter_name: String,
    execution_mode: String,
    gpu: Option<GpuRuntime>,
}

impl SnakeGpuCore {
    fn spawn_one_food(&mut self) {
        let g = self.grid_size as i32;
        let min_dist: i32 = 4;
        let mut far_cells = Vec::new();
        let mut any_free = Vec::new();
        for r in 0..g {
            for c in 0..g {
                let occupied = self.body.iter().any(|&(br, bc)| br == r && bc == c)
                    || self.foods.iter().any(|&(fr, fc)| fr == r && fc == c);
                if !occupied {
                    any_free.push((r, c));
                    let dist = (r - self.head_r).abs() + (c - self.head_c).abs();
                    if dist >= min_dist {
                        far_cells.push((r, c));
                    }
                }
            }
        }
        let candidates = if far_cells.is_empty() { &any_free } else { &far_cells };
        if !candidates.is_empty() {
            let idx = rand::rng().random_range(0..candidates.len());
            self.foods.push(candidates[idx]);
        }
    }

    fn spawn_foods(&mut self) {
        while self.foods.len() < self.num_foods {
            self.spawn_one_food();
        }
    }

    fn gpu_state(&self, action: i32) -> GpuState {
        let mut food_r = [-1i32; NUM_FOODS];
        let mut food_c = [-1i32; NUM_FOODS];
        for (i, &(fr, fc)) in self.foods.iter().enumerate() {
            if i < NUM_FOODS {
                food_r[i] = fr;
                food_c[i] = fc;
            }
        }
        GpuState {
            grid_size: self.grid_size as u32,
            max_steps: self.max_steps as u32,
            step_count: self.step_count as u32,
            direction: self.direction,
            head_r: self.head_r,
            head_c: self.head_c,
            num_foods: self.num_foods as u32,
            done: if self.done { 1 } else { 0 },
            reward: 0.0,
            terminated: 0,
            truncated: 0,
            length: self.body.len() as u32,
            action,
            rng_state: self.rng_state,
            food_r,
            food_c,
            _pad: [0; 2],
        }
    }

    fn gpu_body_vec(&self) -> Vec<[i32; 2]> {
        let max_cells = self.grid_size * self.grid_size;
        let mut out = vec![[0_i32, 0_i32]; max_cells];
        for (i, &(r, c)) in self.body.iter().enumerate() {
            out[i] = [r, c];
        }
        out
    }

    fn apply_gpu_state(&mut self, state: &GpuState) {
        self.step_count = state.step_count as usize;
        self.direction = state.direction;
        self.head_r = state.head_r;
        self.head_c = state.head_c;
        self.foods.clear();
        for i in 0..NUM_FOODS {
            if state.food_r[i] >= 0 && state.food_c[i] >= 0 {
                self.foods.push((state.food_r[i], state.food_c[i]));
            }
        }
        self.done = state.done != 0;
        self.rng_state = state.rng_state;
    }

    fn pixels_u32_to_semantic(pixels: Vec<u32>) -> Vec<u8> {
        // Semantic one-hot channels per cell: [empty, snake, food]
        // Render palette mapping from shader:
        // - empty: 0
        // - snake body: rgb(0,255,0)
        // - head: rgb(255,255,255) -> snake channel
        // - food: rgb(255,0,0)
        let mut out = Vec::with_capacity(pixels.len() * 3);
        for p in pixels {
            let r = (p & 0xFF) as u8;
            let g = ((p >> 8) & 0xFF) as u8;
            let b = ((p >> 16) & 0xFF) as u8;
            if r == 0 && g == 0 && b == 0 {
                out.extend_from_slice(&[1, 0, 0]);
            } else if r == 255 && g == 0 && b == 0 {
                out.extend_from_slice(&[0, 0, 1]);
            } else {
                // snake body or head
                out.extend_from_slice(&[0, 1, 0]);
            }
        }
        out
    }

    fn step_cpu(&mut self, action: i32) -> (Vec<u8>, f32, bool, bool, usize) {
        if self.done {
            return (self.render_flat(), 0.0, true, false, self.body.len());
        }

        let opposite = (self.direction + 2) % 4;
        let a = if action == opposite { self.direction } else { action };
        self.direction = a;

        let (dr, dc) = match a {
            0 => (-1, 0),
            1 => (0, 1),
            2 => (1, 0),
            _ => (0, -1),
        };

        let nr = self.head_r + dr;
        let nc = self.head_c + dc;

        self.step_count += 1;
        let mut reward = -0.01f32;
        let mut terminated = false;
        let mut truncated = false;

        let g = self.grid_size as i32;
        if nr < 0 || nr >= g || nc < 0 || nc >= g {
            self.done = true;
            reward = -1.0;
            terminated = true;
            return (self.render_flat(), reward, terminated, truncated, self.body.len());
        }

        let ate = self.foods.iter().position(|&(fr, fc)| fr == nr && fc == nc);
        let tail = *self.body.last().unwrap();
        let hits_body = self.body.iter().any(|&(r, c)| r == nr && c == nc);
        if hits_body && !((nr, nc) == tail && ate.is_none()) {
            self.done = true;
            reward = -1.0;
            terminated = true;
            return (self.render_flat(), reward, terminated, truncated, self.body.len());
        }

        self.head_r = nr;
        self.head_c = nc;
        self.body.insert(0, (nr, nc));

        if let Some(food_idx) = ate {
            reward = 1.0;
            self.foods.remove(food_idx);
            self.spawn_one_food();
        } else {
            self.body.pop();
        }

        if self.step_count >= self.max_steps {
            self.done = true;
            truncated = true;
        }

        (self.render_flat(), reward, terminated, truncated, self.body.len())
    }

    fn render_flat(&self) -> Vec<u8> {
        // Semantic one-hot channels per cell: [empty, snake, food]
        let n = self.grid_size * self.grid_size * 3;
        let mut out = vec![0u8; n];

        let idx = |r: usize, c: usize, ch: usize, g: usize| -> usize { (r * g + c) * 3 + ch };
        let g = self.grid_size;

        // Mark all empty first
        for r in 0..g {
            for c in 0..g {
                out[idx(r, c, 0, g)] = 1;
            }
        }

        // Mark snake body + head
        for &(r, c) in &self.body {
            if r >= 0 && c >= 0 {
                let (ru, cu) = (r as usize, c as usize);
                if ru < g && cu < g {
                    out[idx(ru, cu, 0, g)] = 0;
                    out[idx(ru, cu, 1, g)] = 1;
                    out[idx(ru, cu, 2, g)] = 0;
                }
            }
        }

        // Mark foods
        for &(fr, fc) in &self.foods {
            if fr >= 0 && fc >= 0 {
                let (ru, cu) = (fr as usize, fc as usize);
                if ru < g && cu < g {
                    out[idx(ru, cu, 0, g)] = 0;
                    out[idx(ru, cu, 1, g)] = 0;
                    out[idx(ru, cu, 2, g)] = 1;
                }
            }
        }

        out
    }
}

#[pymethods]
impl SnakeGpuCore {
    #[new]
    #[pyo3(signature = (grid_size, max_steps, num_foods=5))]
    pub fn new(grid_size: usize, max_steps: usize, num_foods: usize) -> PyResult<Self> {
        let c = (grid_size / 2) as i32;
        let mut core = Self {
            grid_size,
            max_steps,
            num_foods,
            step_count: 0,
            direction: 1,
            head_r: c,
            head_c: c,
            body: vec![(c, c), (c, c - 1), (c, c - 2)],
            foods: Vec::new(),
            done: false,
            rng_state: 12345,
            backend: "None".to_string(),
            adapter_name: "No adapter".to_string(),
            execution_mode: "cpu".to_string(),
            gpu: None,
        };
        core.spawn_foods();

        if let Ok((gpu, backend, adapter_name)) = GpuRuntime::new(grid_size) {
            core.backend = backend;
            core.adapter_name = adapter_name;
            core.execution_mode = "gpu".to_string();
            core.gpu = Some(gpu);
        }

        Ok(core)
    }

    pub fn gpu_info(&self) -> (String, String) {
        (self.backend.clone(), self.adapter_name.clone())
    }

    pub fn execution_mode(&self) -> String {
        self.execution_mode.clone()
    }

    pub fn reset(&mut self) -> Vec<u8> {
        let c = (self.grid_size / 2) as i32;
        self.step_count = 0;
        self.direction = 1;
        self.head_r = c;
        self.head_c = c;
        self.body = vec![(c, c), (c, c - 1), (c, c - 2)];
        self.done = false;
        self.rng_state = rand::rng().random::<u32>() | 1;
        self.foods.clear();
        self.spawn_foods();

        if let Some(gpu) = &self.gpu {
            let state = self.gpu_state(-1);
            let body = self.gpu_body_vec();
            gpu.write_state_and_body(&state, &body);
            if let Ok((_st, pixels, _body)) = gpu.run() {
                return Self::pixels_u32_to_semantic(pixels);
            }
            self.execution_mode = "cpu".to_string();
            self.gpu = None;
        }

        self.render_flat()
    }

    pub fn step(&mut self, action: i32) -> (Vec<u8>, f32, bool, bool, usize) {
        if let Some(gpu) = &self.gpu {
            let state = self.gpu_state(action);
            let body = self.gpu_body_vec();
            gpu.write_state_and_body(&state, &body);
            if let Ok((out_state, pixels, out_body)) = gpu.run() {
                self.apply_gpu_state(&out_state);
                self.body = out_body
                    .into_iter()
                    .take(out_state.length as usize)
                    .map(|xy| (xy[0], xy[1]))
                    .collect();
                return (
                    Self::pixels_u32_to_semantic(pixels),
                    out_state.reward,
                    out_state.terminated != 0,
                    out_state.truncated != 0,
                    out_state.length as usize,
                );
            }
            self.execution_mode = "cpu".to_string();
            self.gpu = None;
        }

        self.step_cpu(action)
    }
}

#[pymodule]
fn snake_gpu_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnakeGpuCore>()?;
    Ok(())
}
