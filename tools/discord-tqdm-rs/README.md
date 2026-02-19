# discord-tqdm-rs

Tiny Rust CLI to post/update a Discord webhook message like a tqdm progress bar.

## Build

```bash
cd tools/discord-tqdm-rs
cargo build --release
```

## Usage

```bash
export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'

# first call creates message
./target/release/discord-tqdm-rs \
  --current 10 --total 100 \
  --label "🧬 Fleet Training" \
  --status "best=294.3 mean=141.2"

# subsequent calls edit same message (state file stores message id)
./target/release/discord-tqdm-rs \
  --current 11 --total 100 \
  --label "🧬 Fleet Training" \
  --status "best=300.2 mean=145.8"
```

Optional:
- `--state-file /path/to/idfile`
- `--width 30`
