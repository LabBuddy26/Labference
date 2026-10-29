from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an OpenPI checkpoint over WebSocket")
    parser.add_argument("--config", required=True, help="OpenPI training config name")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        from openpi.policies import policy_config
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        from openpi.training import config as training_config
    except ModuleNotFoundError as exc:
        raise SystemExit("OpenPI is not installed in this environment") from exc

    train_cfg = training_config.get_config(args.config)
    policy = policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        default_prompt=args.default_prompt,
        pytorch_device=args.device,
    )
    metadata = dict(getattr(policy, "metadata", {}) or {})
    metadata.update({"provider": "openpi", "action_dim": 8})
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
