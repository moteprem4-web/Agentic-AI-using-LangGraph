from mcp import Client

class MultiServerMCPClient:
    def __init__(self, servers: dict):
        self.servers = servers
        self.clients = {}

    async def __aenter__(self):
        for name, cfg in self.servers.items():
            client = Client(**cfg)
            await client.__aenter__()
            self.clients[name] = client
        return self

    async def __aexit__(self, exc_type, exc, tb):
        for client in self.clients.values():
            await client.__aexit__(exc_type, exc, tb)

    async def list_tools(self):
        return {
            name: await client.list_tools()
            for name, client in self.clients.items()
        }

    async def call_tool(self, server: str, tool: str, args: dict):
        return await self.clients[server].call_tool(tool, args)
