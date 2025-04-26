# ksim-kbot

Welcome to the ksim-kbot project!

## Installation

In the root directory, run:

```bash
pip install -e .
```

You'll also need to pull `kscale-assets` by running:

```bash
git submodule update --init --recursive
```

See the [ksim documentation](https://docs.kscale.dev/docs/ksim) for help with training and debugging.

## Deployment

https://gist.github.com/WT-MM/64f5d94d2fee6878cc188ff5691f8b52

## AMP

To generate the reference motion for the AMP task, run (from the root directory):

```bash
ksim-generate-reference -f ksim_kbot/reference_motions/walk_normal.yaml
```

This will generate a `walk_normal_kbot.npz` file in the `ksim_kbot/reference_motions` directory.
