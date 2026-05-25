import asyncio

from dotenv import load_dotenv

load_dotenv()
from agent_workflow import WorkflowInput, run_workflow


async def main():
    complex_prompt = """
    Implement the key findings from the paper on when do stop losses stop losses and try to reproduce the results.
    """

    result = await run_workflow(WorkflowInput(input_as_text=complex_prompt))

    print("\n\nFINAL RESULT")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
