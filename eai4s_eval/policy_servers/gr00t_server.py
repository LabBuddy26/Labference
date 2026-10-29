from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an Isaac-GR00T checkpoint over ZeroMQ")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--no-strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.policy.server_client import PolicyServer
    except ModuleNotFoundError as exc:
        raise SystemExit("Isaac-GR00T is not installed in this environment") from exc

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
        strict=not args.no_strict,
    )
    with PolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
