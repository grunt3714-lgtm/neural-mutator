#!/usr/bin/env python3
"""
Fleet progress aggregator — polls all nodes' /tmp/train_progress.json
and drives a single tqdm_discord bar showing combined fleet progress.

Usage:
    python scripts/fleet_progress.py --nodes 192.168.1.95,192.168.1.99,192.168.1.112,192.168.1.100

Requires env vars:
    TQDM_DISCORD_TOKEN      — Discord bot token
    TQDM_DISCORD_CHANNEL_ID — Channel ID (e.g. #training)

Or pass --token and --channel-id.
"""

import argparse
import json
import subprocess
import time
import os
import sys

from tqdm.contrib.discord import tqdm as tqdm_discord


def read_node_progress(host: str, timeout: int = 5) -> dict | None:
    """SSH to a node and read its progress file."""
    try:
        r = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=3', '-o', 'StrictHostKeyChecking=no',
             f'grunt@{host}', 'cat /tmp/train_progress.json 2>/dev/null'],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Fleet training progress aggregator')
    parser.add_argument('--nodes', required=True,
                        help='Comma-separated node IPs')
    parser.add_argument('--poll-interval', type=float, default=5.0,
                        help='Seconds between polls (default: 5)')
    parser.add_argument('--token', default=None, help='Discord bot token')
    parser.add_argument('--channel-id', default=None, help='Discord channel ID')
    args = parser.parse_args()

    token = args.token or os.environ.get('TQDM_DISCORD_TOKEN')
    channel_id = args.channel_id or os.environ.get('TQDM_DISCORD_CHANNEL_ID')
    if not token or not channel_id:
        print("Error: need --token/TQDM_DISCORD_TOKEN and --channel-id/TQDM_DISCORD_CHANNEL_ID",
              file=sys.stderr)
        sys.exit(1)

    nodes = [n.strip() for n in args.nodes.split(',') if n.strip()]
    print(f"Monitoring {len(nodes)} nodes: {nodes}")

    # Wait for first progress data to determine total
    print("Waiting for training to start on nodes...")
    node_totals = {}
    while not node_totals or len(node_totals) < len(nodes):
        for host in nodes:
            if host not in node_totals:
                p = read_node_progress(host)
                if p and 'total' in p:
                    node_totals[host] = p['total']
        if node_totals:
            # Start once at least one node is reporting
            break
        time.sleep(2)

    total_gens = sum(node_totals.values())
    print(f"Total work: {total_gens} generations across {len(node_totals)} nodes")

    # Create the tqdm_discord bar
    pbar = tqdm_discord(
        total=total_gens,
        token=token,
        channel_id=int(channel_id),
        desc="🐍 Fleet Training",
        unit="gen",
        mininterval=3,
    )

    last_completed = 0

    while True:
        node_data = {}
        for host in nodes:
            p = read_node_progress(host)
            if p:
                node_data[host] = p
                if host not in node_totals and 'total' in p:
                    node_totals[host] = p['total']
                    new_total = sum(node_totals.values())
                    pbar.total = new_total
                    total_gens = new_total

        # Sum completed gens (gen is 0-indexed, so gen+1 = completed)
        completed = sum((d['gen'] + 1) for d in node_data.values())
        all_done = (len(node_data) == len(nodes) and
                    all(d.get('done', False) for d in node_data.values()))

        # Update bar by the delta
        delta = completed - last_completed
        if delta > 0:
            # Build postfix with per-node stats
            fleet_best = max((d.get('best_ever', -999) for d in node_data.values()), default=0)
            postfix = {"fleet_best": f"{fleet_best:+.2f}"}
            for host in nodes:
                d = node_data.get(host)
                if d:
                    name = d.get('node', host.split('.')[-1])
                    icon = "✅" if d.get('done') else "🏃"
                    postfix[f"{icon}{name}"] = f"s{d.get('seed','?')} {d['best_ever']:+.2f}"
            pbar.set_postfix(postfix, refresh=False)
            pbar.update(delta)
            last_completed = completed

        if all_done:
            pbar.close()
            print(f"\nAll nodes done! Fleet best-ever: {fleet_best:+.2f}")
            break

        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
