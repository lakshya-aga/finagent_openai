import asyncio
from dotenv import load_dotenv
load_dotenv()
from agent_workflow import run_workflow, WorkflowInput

async def main():

    prompt = """
    Build a momentum trading strategy on S&P500 stocks.

    Steps:
    - fetch daily prices
    - compute zscore of average returns of last 30 days
    - construct long-short portfolio with top 50 stocks longed and bottom 50 shorted
    - produce asset_returns and asset_weights
    - conduct backtest
    """

    complex_prompt = """
    Implement the key findings from the paper on when do stop losses stop losses and try to reproduce the results.
    """

    result = await run_workflow(
        WorkflowInput(
            input_as_text=complex_prompt
        )
    )

    print("\n\nFINAL RESULT")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())