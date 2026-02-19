use boxdd::{shapes, BodyBuilder, BodyType, ShapeDef, Vec2, World, WorldDef};
use boxdd::types::BodyId;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use reqwest::blocking::Client;
use serde_json::json;
use std::sync::{Arc, Mutex};
use tch::{no_grad_guard, CModule, Kind, Tensor};

struct CustomLunarLander {
    world: World,
    lander_id: BodyId,
    steps: usize,
    max_steps: usize,
    left_contact: f32,
    right_contact: f32,
}

impl CustomLunarLander {
    fn new(max_steps: usize) -> Self {
        let wdef = WorldDef::builder().gravity(Vec2::new(0.0, -9.8)).build();
        let mut world = World::new(wdef).expect("create world failed");

        let ground = world.create_body_id(
            BodyBuilder::new()
                .body_type(BodyType::Static)
                .position([0.0, -0.2])
                .build(),
        );
        let sdef = ShapeDef::builder().build();
        let ground_poly = shapes::box_polygon(2.0, 0.2);
        world.create_polygon_shape_for(ground, &sdef, &ground_poly);

        let lander_id = world.create_body_id(
            BodyBuilder::new()
                .body_type(BodyType::Dynamic)
                .position([0.0, 1.0])
                .linear_damping(0.1)
                .angular_damping(0.1)
                .build(),
        );
        let lander_poly = shapes::box_polygon(0.06, 0.09);
        let sdef_lander = ShapeDef::builder().density(5.0).build();
        world.create_polygon_shape_for(lander_id, &sdef_lander, &lander_poly);

        Self {
            world,
            lander_id,
            steps: 0,
            max_steps,
            left_contact: 0.0,
            right_contact: 0.0,
        }
    }

    fn reset(&mut self, rng: &mut StdRng) -> [f32; 8] {
        let x = rng.gen_range(-0.1..0.1);
        let y = 1.0 + rng.gen_range(-0.05..0.05);
        let vx = rng.gen_range(-0.2..0.2);
        let vy = rng.gen_range(-0.2..0.0);
        let theta = rng.gen_range(-0.05..0.05);

        {
            let mut b = self.world.body(self.lander_id).expect("lander body missing");
            b.set_position_and_rotation([x, y], theta);
            b.set_linear_velocity([vx, vy]);
            b.set_angular_velocity(0.0);
        }

        self.steps = 0;
        self.left_contact = 0.0;
        self.right_contact = 0.0;
        self.obs()
    }

    fn obs(&mut self) -> [f32; 8] {
        let b = self.world.body(self.lander_id).expect("lander body missing");
        let p = b.position();
        let v = b.linear_velocity();
        let w = b.angular_velocity();
        let t = b.transform();
        let theta = t.q.s.atan2(t.q.c);

        // Approximate leg contacts by hull corners touching ground plane.
        self.left_contact = if p.y - 0.09 <= 0.0 && p.x < 0.0 { 1.0 } else { 0.0 };
        self.right_contact = if p.y - 0.09 <= 0.0 && p.x >= 0.0 { 1.0 } else { 0.0 };

        [p.x, p.y, v.x, v.y, theta, w, self.left_contact, self.right_contact]
    }

    fn step(&mut self, action: i64) -> ([f32; 8], f32, bool) {
        // actions: 0=noop, 1=left thruster, 2=main thruster, 3=right thruster
        let main = if action == 2 { 18.0 } else { 0.0 };
        let side = if action == 1 { -6.0 } else if action == 3 { 6.0 } else { 0.0 };

        {
            let mut b = self.world.body(self.lander_id).expect("lander body missing");
            if main != 0.0 {
                b.apply_force_to_center([0.0, main], true);
            }
            if side != 0.0 {
                b.apply_torque(side * 0.02, true);
                b.apply_force_to_center([side, 0.0], true);
            }
        }

        self.world.step(1.0 / 50.0, 8);
        self.steps += 1;

        let obs = self.obs();
        let x = obs[0];
        let y = obs[1];
        let vx = obs[2];
        let vy = obs[3];
        let theta = obs[4];

        let mut done = false;
        let mut reward = -0.03;

        if y <= 0.0 {
            let soft = vy.abs() < 0.6 && theta.abs() < 0.2 && x.abs() < 0.15;
            done = true;
            reward += if soft { 120.0 } else { -100.0 };
        }
        if x.abs() > 1.2 || self.steps >= self.max_steps {
            done = true;
            reward -= 25.0;
        }

        // Gym-like shaping terms
        reward += 1.2 * (1.0 - x.abs());
        reward += 1.0 * (1.0 - y.min(1.0));
        reward -= 0.2 * vx.abs();
        reward -= 0.3 * vy.abs();
        reward -= 0.2 * theta.abs();
        reward += 0.1 * (self.left_contact + self.right_contact);
        if action == 2 {
            reward -= 0.30;
        }
        if action == 1 || action == 3 {
            reward -= 0.03;
        }

        (obs, reward, done)
    }
}

fn policy_action(model: &CModule, obs: [f32; 8]) -> i64 {
    let _ng = no_grad_guard();
    let t = Tensor::from_slice(&obs)
        .to_kind(Kind::Float)
        .reshape([1, 8]);
    let out = model
        .forward_ts(&[t])
        .expect("forward failed")
        .reshape([4]);
    out.argmax(0, false).int64_value(&[])
}

#[pyfunction(signature = (mutator_path, flat, split_idx, seed=None))]
fn mutator_mutate(
    mutator_path: String,
    flat: Vec<f32>,
    split_idx: i64,
    seed: Option<i64>,
) -> PyResult<Vec<f32>> {
    if let Some(s) = seed {
        tch::manual_seed(s);
    }
    let model = CModule::load(mutator_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load mutator failed: {e}")))?;
    let _ng = no_grad_guard();
    let w = Tensor::from_slice(&flat).to_kind(Kind::Float);
    let s = Tensor::from(split_idx);
    let out = model
        .forward_ts(&[w, s])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("mutator forward failed: {e}")))?;
    let flat_out = out.reshape([-1]).to_kind(Kind::Float);
    let n = flat_out.numel();
    let mut v = vec![0f32; n];
    flat_out.copy_data(&mut v, n);
    Ok(v)
}

#[pyfunction]
fn compat_score(compat_path: String, genome_a: Vec<f32>, genome_b: Vec<f32>) -> PyResult<f32> {
    let model = CModule::load(compat_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load compat failed: {e}")))?;
    let _ng = no_grad_guard();
    let a = Tensor::from_slice(&genome_a).to_kind(Kind::Float);
    let b = Tensor::from_slice(&genome_b).to_kind(Kind::Float);
    let out = model
        .forward_ts(&[a, b])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("compat forward failed: {e}")))?;
    Ok(out.double_value(&[]) as f32)
}

#[pyfunction(signature = (model_path, episodes, max_steps, threads, seed, webhook=None, progress_every=None))]
fn evaluate_model_threaded(
    model_path: String,
    episodes: usize,
    max_steps: usize,
    threads: usize,
    seed: u64,
    webhook: Option<String>,
    progress_every: Option<usize>,
) -> PyResult<f32> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let model = Arc::new(
        CModule::load(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load model failed: {e}")))?
    );

    let scores = Arc::new(Mutex::new(vec![0.0_f32; episodes]));
    let webhook_client = webhook.as_ref().map(|_| Client::new());
    let every = progress_every.unwrap_or(0);

    pool.install(|| {
        (0..episodes).into_par_iter().for_each(|ep| {
            let mut rng = StdRng::seed_from_u64(seed + ep as u64);
            let mut env = CustomLunarLander::new(max_steps);
            let mut obs = env.reset(&mut rng);
            let mut total = 0.0_f32;
            loop {
                let action = policy_action(&model, obs);
                let (next_obs, r, done) = env.step(action);
                total += r;
                obs = next_obs;
                if done {
                    break;
                }
            }
            {
                let mut lock = scores.lock().expect("scores mutex poisoned");
                lock[ep] = total;
            }

            if every > 0 && (ep + 1) % every == 0 {
                if let (Some(url), Some(client)) = (&webhook, &webhook_client) {
                    let done = ep + 1;
                    let pct = (done as f64 / episodes as f64) * 100.0;
                    let _ = client
                        .post(url)
                        .json(&json!({
                            "content": format!(
                                "🦀 Rust Lunar eval: {}/{} ({:.1}%)",
                                done, episodes, pct
                            )
                        }))
                        .send();
                }
            }
        });
    });

    let mean = {
        let lock = scores.lock().expect("scores mutex poisoned");
        lock.iter().sum::<f32>() / episodes as f32
    };

    Ok(mean)
}

#[pyfunction]
fn benchmark_thread_scaling(model_path: String, episodes: usize, max_steps: usize, seed: u64) -> PyResult<Vec<(usize, f32)>> {
    let mut out = Vec::new();
    for t in [1usize, 2, 4, 8] {
        let score = evaluate_model_threaded(model_path.clone(), episodes, max_steps, t, seed, None, None)?;
        out.push((t, score));
    }
    Ok(out)
}

#[pymodule]
fn lunar_torch_eval_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_model_threaded, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_thread_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(mutator_mutate, m)?)?;
    m.add_function(wrap_pyfunction!(compat_score, m)?)?;
    Ok(())
}
