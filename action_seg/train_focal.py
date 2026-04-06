#!/usr/bin/env python3

"""Scenario-specific training entrypoint that locks focal loss to gamma=2."""

from __future__ import annotations

import sys


def enforce_focal_args(args):
    normalized_args = []
    index = 0

    while index < len(args):
        argument = args[index]
        if argument == "--focal_gamma":
            normalized_args.extend(["--focal_gamma", "2.0"])
            index += 2
            continue

        normalized_args.append(argument)
        index += 1

    if "--focal_gamma" not in normalized_args:
        normalized_args.extend(["--focal_gamma", "2.0"])
    if "--use_focal_loss" not in normalized_args:
        normalized_args.append("--use_focal_loss")

    return normalized_args


def main():
    sys.argv = [sys.argv[0], *enforce_focal_args(sys.argv[1:])]

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
