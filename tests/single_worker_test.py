import asyncio

from api_server.server import Server
from model.util import benchmark_memory

prompts = [
    "Write a hello world program in python. Just give me the code.",
    "Who is the best soccer player of all time?",
    "Who is inside of the duolingo suit?",
    "Hables espanol? If so, prove it with the most creative paragraph you can think of."
]

# how much benchmarking output to see, higher = more and 0 = none
BENCHMARK = 1

async def main():
    device = "mps"

    if BENCHMARK >= 1:
        benchmark_memory("before starting server", device)
    server = Server(skip_load=False, debug=1, benchmark=BENCHMARK, device="mps")
    if BENCHMARK >= 1:
        benchmark_memory("after starting server", device)

    tasks = [
            asyncio.create_task(server.generate(prompt, top_k=1)) for prompt in prompts
        ]

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"Prompt {i}")
        if BENCHMARK >= 1:
            benchmark_memory(f"after prompt {i}", device)
            print(f"benchmark stats: TTFT: {result['benchmarks']['TTFT']}, TBT: {result['benchmarks']['TBT']}")
        print("text:")
        print(result['text'])


if __name__ == "__main__":
    asyncio.run(main())
