import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i+1]) for i in range(4)]
biases = jnp.zeros((5,))
weights = jnp.ones((4,)) * 0.5
beta = jnp.array(1.0)
model = IsingEBM(nodes, edges, biases, weights, beta)

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
# Energy of an Ising configuration: -J * sum(s_i * s_{i+1}) - h * sum(s_i)
# Here J = 0.5, h = 0
spins = jnp.where(samples[0], 1.0, -1.0)
pair_products = spins[:, :-1] * spins[:, 1:]
energies = -0.5 * jnp.sum(pair_products, axis=1)
avg_energy = jnp.mean(energies)
print("Average energy:", avg_energy)
print("Min energy seen:", jnp.min(energies))
print("Max energy seen:", jnp.max(energies))