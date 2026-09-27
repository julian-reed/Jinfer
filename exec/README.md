## Execution

Handles the actually execution of the forward pass, given a batch to process.

Mainly made up of a worker, which will be spawned onto each GPU (in a distributed setup).
