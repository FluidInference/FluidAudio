# MCP

The repo is indexed by the DeepWiki MCP server, so your coding tools can access documentation programmatically.

## VS Code / MCP Client Configuration

Add this to your MCP client configuration file:

```json
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/mcp"
    }
  }
}
```

## Claude Code (CLI)

```bash
claude mcp add -s user -t http deepwiki https://mcp.deepwiki.com/mcp
```

