# Deep MTR

`deep_mtr` is buildable migration scaffolding for Motion Transformer integration. It defines a
lifecycle node that accepts semantic `deep_msgs/msg/MtrScene` requests, but it does not run MTR and
does not publish `deep_msgs/msg/MtrPredictionArray` results.

The node is deliberately safe to launch without a model. While active, it logs a throttled warning
for incoming scene requests and produces no fabricated predictions.

## Topics

- Input: `/mtr/scenes`
- Reserved output: `/mtr/predictions`

Both topic names are configurable through `config/deep_mtr.yaml`.

## Lifecycle

The node creates its publisher during configuration and subscribes to scenes only while active.
Configure and activate it with standard ROS 2 lifecycle commands when testing the scaffold.

Working inference is intentionally excluded. See `DEVELOPING.md` for the implementation ownership
boundary.
