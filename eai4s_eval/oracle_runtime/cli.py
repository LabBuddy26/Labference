"""Protocol preflight utilities that do not start Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import check_capabilities, load_intervention_spec, load_policy_profile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Labference policy and Oracle protocol preflight")
    parser.add_argument("--policy-profile", required=True)
    parser.add_argument("--intervention-spec", required=True)
    parser.add_argument("--server-metadata", help="captured metadata JSON from the policy service")
    args = parser.parse_args(argv)
    profile = load_policy_profile(args.policy_profile)
    intervention = load_intervention_spec(args.intervention_spec)
    metadata = json.loads(Path(args.server_metadata).read_text()) if args.server_metadata else None
    result = check_capabilities(profile, intervention, metadata)
    print(json.dumps({"compatible": result.compatible, "errors": result.errors,
                      "warnings": result.warnings, "verified_from_server": result.verified_from_server,
                      "policy_profile": profile.public(),
                      "intervention": intervention.__dict__}, indent=2))
    if not result.compatible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
