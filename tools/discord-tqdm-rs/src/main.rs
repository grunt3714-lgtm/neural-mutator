use anyhow::{anyhow, Result};
use clap::Parser;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "discord-tqdm-rs")]
struct Args {
    /// Discord webhook URL
    #[arg(long)]
    webhook: String,

    /// Current step
    #[arg(long)]
    current: u64,

    /// Total steps
    #[arg(long)]
    total: u64,

    /// Prefix text
    #[arg(long, default_value = "Progress")]
    label: String,

    /// Optional status text suffix
    #[arg(long, default_value = "")]
    status: String,

    /// Width of bar
    #[arg(long, default_value_t = 20)]
    width: usize,

    /// File to store/reuse Discord message id
    #[arg(long, default_value = "/tmp/discord_tqdm_message_id")]
    state_file: PathBuf,
}

#[derive(Deserialize)]
struct DiscordMessageResp {
    id: String,
}

fn make_bar(cur: u64, total: u64, width: usize) -> String {
    if total == 0 {
        return "░".repeat(width);
    }
    let frac = (cur as f64 / total as f64).clamp(0.0, 1.0);
    let filled = (frac * width as f64).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width.saturating_sub(filled)))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let client = Client::new();

    let pct = if args.total == 0 {
        0.0
    } else {
        (args.current as f64 / args.total as f64) * 100.0
    };
    let bar = make_bar(args.current, args.total, args.width);
    let content = if args.status.is_empty() {
        format!("{}\n`{}` `{}/{} ({:.1}%)`", args.label, bar, args.current, args.total, pct)
    } else {
        format!(
            "{}\n`{}` `{}/{} ({:.1}%)`\n{}",
            args.label, bar, args.current, args.total, pct, args.status
        )
    };

    let message_id = fs::read_to_string(&args.state_file).ok().map(|s| s.trim().to_string());

    if let Some(mid) = message_id.filter(|s| !s.is_empty()) {
        let url = format!("{}/messages/{}", args.webhook, mid);
        let resp = client
            .patch(url)
            .query(&[("wait", "true")])
            .json(&json!({ "content": content }))
            .send()?;

        if resp.status().is_success() {
            return Ok(());
        }
    }

    let resp = client
        .post(&args.webhook)
        .query(&[("wait", "true")])
        .json(&json!({ "content": content }))
        .send()?;

    if !resp.status().is_success() {
        return Err(anyhow!("discord webhook post failed: {}", resp.status()));
    }

    let body: DiscordMessageResp = resp.json()?;
    fs::write(&args.state_file, body.id)?;
    Ok(())
}
