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
    server = Server(debug=1, benchmark=BENCHMARK, device="mps")
    if BENCHMARK >= 1:
        benchmark_memory("after starting server", device)

    for i in range(len(prompts)):
        prompt = prompts[i]
        print(f"serving prompt {prompt}")

        # archived sampling params: min_p=0.1
        out_dict = await server.generate(prompt, top_k=1)
        if BENCHMARK >= 1:
            benchmark_memory(f"after prompt {i}", device)
            print(f"benchmark stats: TTFT: {out_dict['benchmarks']['TTFT']}, TBT: {out_dict['benchmarks']['TBT']}")
        print("text:")
        print(out_dict['text'])


if __name__ == "__main__":
    asyncio.run(main())
