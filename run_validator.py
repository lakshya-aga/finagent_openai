import asyncio
import sys
from dotenv import load_dotenv
load_dotenv()
from agent_workflow import validatorandfixingagent, RunConfig
from agents import Runner
from agents.mcp import MCPServerSse, MCPServerSseParams, MCPServerManager, MCPServerStreamableHttp, MCPServerStreamableHttpParams

async def main(notebook_path: str):
    _fruit = MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(url="http://localhost:8090/mcp/"),
        name="fruit_thrower",
        tool_filter={"allowed_tool_names": ["search_code", "get_unit_source", "list_modules", "get_module_summary", "index_repository", "get_index_stats", "generate_function"]},
        require_approval="never",
    )
    _data = MCPServerSse(
        params=MCPServerSseParams(url="http://localhost:8000/sse"),
        name="data_mcp",
        tool_filter={"allowed_tool_names": ["search_tools", "get_tool_doc", "list_all_tools", "request_data_source"]},
        require_approval="never",
    )
    async with MCPServerManager([_fruit, _data]) as mgr:
      _validator = validatorandfixingagent.clone(mcp_servers=mgr.active_servers)
      result = await Runner.run(
        _validator,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Validate and fix the notebook at: {notebook_path}"
                    }
                ]
            }
        ],
        run_config=RunConfig(),
        max_turns=20
    )
    print("\n\nVALIDATOR RESULT")
    print(result.final_output)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/notebook_1.ipynb"
    asyncio.run(main(path))
