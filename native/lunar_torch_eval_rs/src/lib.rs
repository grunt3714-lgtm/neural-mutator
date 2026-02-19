use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;
use tch::{no_grad_guard, CModule, Kind, Tensor};

#[derive(Clone)]
struct CustomLunarLander {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    theta: f32,
    omega: f32,
    left_contact: f32,
    right_contact: f32,
    steps: usize,
    max_steps: usize,
}

impl CustomLunarLander {
    fn new(max_steps: usize) -> Self {
        Self {
            x: 0.0,
            y: 1.0,
            vx: 0.0,
            vy: 0.0,
            theta: 0.0,
            omega: 0.0,
            left_contact: 0.0,
            right_contact: 0.0,
            steps: 0,
            max_steps,
        }
    }

    fn reset(&mut self, rng: &mut StdRng) -> [f32; 8] {
        self.x = rng.gen_range(-0.1..0.1);
        self.y = 1.0 + rng.gen_range(-0.05..0.05);
        self.vx = rng.gen_range(-0.02..0.02);
        self.vy = rng.gen_range(-0.02..0.0);
        self.theta = rng.gen_range(-0.05..0.05);
        self.omega = 0.0;
        self.left_contact = 0.0;
        self.right_contact = 0.0;
        self.steps = 0;
        self.obs()
    }

    fn obs(&self) -> [f32; 8] {
        [
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.theta,
            self.omega,
            self.left_contact,
            self.right_contact,
        ]
    }

    fn step(&mut self, action: i64) -> ([f32; 8], f32, bool) {
        // actions: 0=noop, 1=left thruster, 2=main thruster, 3=right thruster
        let g = -0.0035_f32;
        let main = if action == 2 { 0.0065 } else { 0.0 };
        let side = if action == 1 { -0.0025 } else if action == 3 { 0.0025 } else { 0.0 };

        self.vx += side;
        self.vy += g + main;
        self.omega += side * 0.25;
        self.theta += self.omega;
        self.x += self.vx;
        self.y += self.vy;
        self.steps += 1;

        let mut done = false;
        let mut reward = -0.03;

        if self.y <= 0.0 {
            self.y = 0.0;
            let soft = self.vy.abs() < 0.06 && self.theta.abs() < 0.2 && self.x.abs() < 0.15;
            self.left_contact = 1.0;
            self.right_contact = 1.0;
            done = true;
            reward += if soft { 120.0 } else { -100.0 };
        }

        if self.x.abs() > 1.2 || self.steps >= self.max_steps {
            done = true;
            reward -= 25.0;
        }

        // shaping
        reward += 1.2 * (1.0 - self.x.abs());
        reward += 1.0 * (1.0 - self.y.min(1.0));
        reward -= 0.2 * self.vx.abs();
        reward -= 0.3 * self.vy.abs();
        reward -= 0.2 * self.theta.abs();

        (self.obs(), reward, done)
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

#[pyfunction]
fn evaluate_model_threaded(model_path: String, episodes: usize, max_steps: usize, threads: usize, seed: u64) -> PyResult<f32> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let model = Arc::new(
        CModule::load(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load model failed: {e}")))?
    );

    let mean = pool.install(|| {
        (0..episodes)
            .into_par_iter()
            .map(|ep| {
                let mut rng = StdRng::seed_from_u64(seed + ep as u64);
                let mut env = CustomLunarLander::new(max_steps);
                let mut obs = env.reset(&mut rng);
                let mut total = 0.0_f32;
                loop {
                    let action = policy_action(&model, obs);
                    let (next_obs, r, done) = env.step(action);
                    total += r;
                    obs = next_obs;
                    if done { break; }
                }
                total
            })
            .sum::<f32>() / episodes as f32
    });

    Ok(mean)
}

#[pyfunction]
fn benchmark_thread_scaling(model_path: String, episodes: usize, max_steps: usize, seed: u64) -> PyResult<Vec<(usize, f32)>> {
    let mut out = Vec::new();
    for t in [1usize, 2, 4, 8] {
        let score = evaluate_model_threaded(model_path.clone(), episodes, max_steps, t, seed)?;
        out.push((t, score));
    }
    Ok(out)
}

#[pymodule]
fn lunar_torch_eval_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_model_threaded, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_thread_scaling, m)?)?;
    Ok(())
}
