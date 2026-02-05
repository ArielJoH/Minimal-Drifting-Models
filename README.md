# Minimal-Drifting-Models
 A minimal implementation of Drifting Models for 2D toy data. Unlike diffusion/flow models that iterate at inference, drifting models evolve the pushforward distribution during training and generate in a single forward pass (1-NFE). The drifting field V governs sample movement: V -> 0 as generated matches data.
