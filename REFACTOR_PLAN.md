# Open WebUI Chat Processing Refactor Plan

## Executive Summary

This document outlines the refactor plan to transform Open WebUI's chat processing from a hardcoded feature pipeline to a dynamic, LLM-driven agent system where external features (memory, web search, image generation, code interpreter, RAG) are treated as invocable tools/agents.

## Current Architecture Analysis

### Current Flow
```
Chat Request → process_chat_payload() → Fixed Feature Pipeline → process_chat_response()
```

### Current Feature Pipeline (Fixed Order)
1. **Pipeline Inlet Filter** - External pipeline processing
2. **Filter Inlet Functions** - Custom function filters  
3. **Memory Handler** - Query user memory, inject context
4. **Web Search Handler** - Generate queries, search web, add files
5. **Image Generation Handler** - Generate images from prompts
6. **Code Interpreter Setup** - Add system prompt for code execution
7. **Tools Function Calling** - Process external tools
8. **Files/RAG Handler** - Process documents, retrieve context
9. **Context Injection** - Add retrieved context to messages

### Current Implementation Locations
- **Entry Point**: [`main.py:394`](backend/open_webui/main.py:394) - `chat_completion()`
- **Core Processing**: [`main.py:495`](backend/open_webui/main.py:495) - `process_chat()`
- **Payload Processing**: [`middleware.py:753`](backend/open_webui/utils/middleware.py:753) - `process_chat_payload()`
- **Response Processing**: [`middleware.py:1072`](backend/open_webui/utils/middleware.py:1072) - `process_chat_response()`

### Current Feature Handlers
| Feature | Handler Function | Location | Input/Output |
|---------|------------------|----------|--------------|
| Memory | `chat_memory_handler()` | `middleware.py:324` | Modifies `form_data["messages"]` |
| Web Search | `chat_web_search_handler()` | `middleware.py:363` | Adds to `form_data["files"]` |
| Image Generation | `chat_image_generation_handler()` | `middleware.py:525` | Modifies `form_data["messages"]` |
| Code Interpreter | Inline setup | `middleware.py:907` | Modifies `form_data["messages"]` |
| RAG/Files | `chat_completion_files_handler()` | `middleware.py:626` | Returns sources for context |
| Tools | `chat_completion_tools_handler()` | `middleware.py:128` | Processes tool calls |

### Current Issues
1. **Fixed Pipeline Order**: Features always execute in predetermined sequence
2. **Mixed Interfaces**: Inconsistent input/output patterns across handlers
3. **No Dynamic Discovery**: LLM cannot choose which features to use
4. **Tight Coupling**: Features directly modify form_data structure
5. **Limited Composability**: Cannot combine features in flexible ways

## Target Architecture

### New Flow
```
Chat Request → Load Available Agents → Add to tools_dict → LLM Tool Selection → Agent Execution → Response Generation
```

### Agent-Based Design Principles
1. **Unified Interface**: All features implement consistent agent interface
2. **Dynamic Discovery**: LLM decides which agents to invoke
3. **Composable**: Agents can be combined in any order
4. **Extensible**: Easy to add new agents
5. **Decoupled**: Agents return structured data, don't modify form_data

### Core Interfaces

#### Base Agent Interface
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

class AgentResult(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    files: List[str] = []
    sources: List[Dict] = []

class ChatAgent(ABC):
    @abstractmethod
    async def execute(self, params: Dict[str, Any], context: 'ChatContext') -> AgentResult:
        """Execute the agent with given parameters"""
        pass
    
    @abstractmethod
    def get_tool_spec(self) -> Dict[str, Any]:
        """Return OpenAI tool specification for this agent"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for tool calling"""
        pass
```

#### Chat Context
```python
class ChatContext(BaseModel):
    user: UserModel
    request: Request
    messages: List[Dict]
    metadata: Dict[str, Any]
    app_state: Any
    event_emitter: Callable
    event_caller: Callable
```

## Agent Implementations

### 1. Memory Agent
```python
class MemoryAgent(ChatAgent):
    name = "query_memory"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "query_memory",
                "description": "Query user's memory for relevant past conversations and context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search in user's memory"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of memory items to retrieve",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        query = params.get("query", "")
        k = params.get("k", 3)
        
        # Use existing memory query logic
        results = await query_memory(
            context.request, 
            QueryMemoryForm(content=query, k=k), 
            context.user
        )
        
        user_context = ""
        if results and hasattr(results, "documents"):
            for doc_idx, doc in enumerate(results.documents[0]):
                created_at = "Unknown Date"
                if results.metadatas[0][doc_idx].get("created_at"):
                    timestamp = results.metadatas[0][doc_idx]["created_at"]
                    created_at = time.strftime("%Y-%m-%d", time.localtime(timestamp))
                user_context += f"{doc_idx + 1}. [{created_at}] {doc}\n"
        
        return AgentResult(
            content=f"Retrieved memory context:\n{user_context}",
            metadata={"memory_results": len(results.documents[0]) if results else 0}
        )
```

### 2. Web Search Agent
```python
class WebSearchAgent(ChatAgent):
    name = "search_web"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries to execute"
                        }
                    },
                    "required": ["queries"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        queries = params.get("queries", [])
        
        # Emit status
        await context.event_emitter({
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "Searching the web",
                "done": False,
            },
        })
        
        # Use existing web search logic
        results = await process_web_search(
            context.request,
            SearchForm(queries=queries),
            user=context.user,
        )
        
        files = []
        if results:
            if results.get("collection_names"):
                for col_idx, collection_name in enumerate(results.get("collection_names")):
                    files.append({
                        "collection_name": collection_name,
                        "name": ", ".join(queries),
                        "type": "web_search",
                        "urls": results["filenames"],
                        "queries": queries,
                    })
            elif results.get("docs"):
                files.append({
                    "docs": results["docs"],
                    "name": ", ".join(queries),
                    "type": "web_search",
                    "urls": results["filenames"],
                    "queries": queries,
                })
        
        await context.event_emitter({
            "type": "status",
            "data": {
                "action": "web_search",
                "description": f"Searched {len(results.get('filenames', []))} sites",
                "urls": results.get("filenames", []),
                "done": True,
            },
        })
        
        return AgentResult(
            content=f"Web search completed for queries: {', '.join(queries)}",
            files=files,
            metadata={"urls": results.get("filenames", []), "queries": queries}
        )
```

### 3. Image Generation Agent
```python
class ImageGenerationAgent(ChatAgent):
    name = "generate_image"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on a text prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Text prompt describing the image to generate"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        prompt = params.get("prompt", "")
        
        await context.event_emitter({
            "type": "status",
            "data": {"description": "Generating an image", "done": False},
        })
        
        try:
            images = await image_generations(
                request=context.request,
                form_data=GenerateImageForm(prompt=prompt),
                user=context.user,
            )
            
            await context.event_emitter({
                "type": "status",
                "data": {"description": "Generated an image", "done": True},
            })
            
            await context.event_emitter({
                "type": "files",
                "data": {
                    "files": [
                        {"type": "image", "url": image["url"]}
                        for image in images
                    ]
                },
            })
            
            return AgentResult(
                content="Image has been generated successfully",
                metadata={"images": [img["url"] for img in images]}
            )
            
        except Exception as e:
            await context.event_emitter({
                "type": "status",
                "data": {
                    "description": "An error occurred while generating an image",
                    "done": True,
                },
            })
            
            return AgentResult(
                content="Unable to generate an image, an error occurred",
                metadata={"error": str(e)}
            )
```

### 4. Code Interpreter Agent
```python
class CodeInterpreterAgent(ChatAgent):
    name = "execute_code"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute Python code and return the results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (currently only 'python' supported)",
                            "default": "python"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        code = params.get("code", "")
        language = params.get("language", "python")
        
        if language != "python":
            return AgentResult(
                content="Only Python code execution is currently supported",
                metadata={"error": "Unsupported language"}
            )
        
        try:
            # Use existing code execution logic
            if context.app_state.config.CODE_INTERPRETER_ENGINE == "pyodide":
                output = await context.event_caller({
                    "type": "execute:python",
                    "data": {
                        "id": str(uuid4()),
                        "code": code,
                        "session_id": context.metadata.get("session_id", None),
                    },
                })
            elif context.app_state.config.CODE_INTERPRETER_ENGINE == "jupyter":
                output = await execute_code_jupyter(
                    context.app_state.config.CODE_INTERPRETER_JUPYTER_URL,
                    code,
                    # ... other jupyter params
                )
            else:
                output = {"stdout": "Code interpreter engine not configured."}
            
            result_content = ""
            if isinstance(output, dict):
                stdout = output.get("stdout", "")
                stderr = output.get("stderr", "")
                result = output.get("result", "")
                
                if stdout:
                    result_content += f"Output:\n{stdout}\n"
                if stderr:
                    result_content += f"Errors:\n{stderr}\n"
                if result:
                    result_content += f"Result:\n{result}\n"
            
            return AgentResult(
                content=f"Code executed successfully:\n```python\n{code}\n```\n\n{result_content}",
                metadata={"output": output}
            )
            
        except Exception as e:
            return AgentResult(
                content=f"Code execution failed: {str(e)}",
                metadata={"error": str(e)}
            )
```

### 5. RAG Agent
```python
class RAGAgent(ChatAgent):
    name = "query_documents"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "query_documents",
                "description": "Query uploaded documents and knowledge base for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search in documents"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of relevant chunks to retrieve",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, params: Dict[str, Any], context: ChatContext) -> AgentResult:
        query = params.get("query", "")
        k = params.get("k", 5)
        
        files = context.metadata.get("files", [])
        if not files:
            return AgentResult(
                content="No documents available to query",
                metadata={"files_count": 0}
            )
        
        # Use existing RAG logic
        sources = get_sources_from_items(
            request=context.request,
            items=files,
            queries=[query],
            embedding_function=lambda q, prefix: context.app_state.EMBEDDING_FUNCTION(
                q, prefix=prefix, user=context.user
            ),
            k=k,
            # ... other RAG parameters
        )
        
        context_string = ""
        for source in sources:
            if "document" in source:
                for document_text, document_metadata in zip(
                    source["document"], source["metadata"]
                ):
                    source_name = source.get("source", {}).get("name", "Unknown")
                    context_string += f"Source: {source_name}\n{document_text}\n\n"
        
        return AgentResult(
            content=f"Retrieved relevant information:\n\n{context_string}",
            sources=sources,
            metadata={"query": query, "sources_count": len(sources)}
        )
```

## Migration Strategy

### Phase 1: Foundation (Week 1-2)
- [ ] Create agent base classes and interfaces
- [ ] Implement agent registry system
- [ ] Create chat context wrapper
- [ ] Add agent loading mechanism to `process_chat_payload()`
- [ ] Keep existing handlers as fallback

### Phase 2: Agent Implementation (Week 3-4)
- [ ] Implement Memory Agent
- [ ] Implement Web Search Agent  
- [ ] Implement Image Generation Agent
- [ ] Implement Code Interpreter Agent
- [ ] Implement RAG Agent
- [ ] Add agents to tools_dict in parallel with existing handlers

### Phase 3: Integration Testing (Week 5)
- [ ] Test agent-based flow alongside existing flow
- [ ] Add feature flag to switch between old/new systems
- [ ] Performance testing and optimization
- [ ] User acceptance testing

### Phase 4: Migration (Week 6)
- [ ] Default to agent-based system
- [ ] Remove old hardcoded feature handlers
- [ ] Clean up deprecated code
- [ ] Update documentation

### Phase 5: Enhancement (Week 7+)
- [ ] Add agent composition capabilities
- [ ] Implement agent dependencies
- [ ] Add agent configuration UI
- [ ] Performance monitoring and analytics

## Implementation Details

### Agent Registry
```python
class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, ChatAgent] = {}
    
    def register(self, agent: ChatAgent):
        self._agents[agent.name] = agent
    
    def get_all_tools(self) -> Dict[str, Dict]:
        return {
            name: {
                "spec": agent.get_tool_spec(),
                "callable": agent.execute,
                "agent": agent
            }
            for name, agent in self._agents.items()
        }
    
    def get_agent(self, name: str) -> Optional[ChatAgent]:
        return self._agents.get(name)

# Global registry
agent_registry = AgentRegistry()
```

### Modified process_chat_payload()
```python
async def process_chat_payload(request, form_data, user, metadata, model):
    # ... existing setup code ...
    
    # Load agents into tools_dict
    agent_tools = agent_registry.get_all_tools()
    
    # Create chat context
    context = ChatContext(
        user=user,
        request=request,
        messages=form_data["messages"],
        metadata=metadata,
        app_state=request.app.state,
        event_emitter=event_emitter,
        event_caller=event_call
    )
    
    # Add agent tools to existing tools_dict
    tools_dict.update(agent_tools)
    
    # Remove old hardcoded feature handling
    # features = form_data.pop("features", None)
    # if features:
    #     # Old feature handlers removed
    
    # Let existing chat_completion_tools_handler manage everything
    if tools_dict:
        form_data, flags = await chat_completion_tools_handler(
            request, form_data, extra_params, user, models, tools_dict
        )
        sources.extend(flags.get("sources", []))
    
    # ... rest of existing code ...
```

## Testing Strategy

### Unit Tests
- [ ] Test each agent implementation independently
- [ ] Test agent registry functionality
- [ ] Test chat context creation and usage
- [ ] Test tool specification generation

### Integration Tests
- [ ] Test agent execution within chat flow
- [ ] Test multiple agent combinations
- [ ] Test error handling and fallbacks
- [ ] Test performance under load

### End-to-End Tests
- [ ] Test complete chat flows with agents
- [ ] Test UI integration with new agent system
- [ ] Test backward compatibility during migration
- [ ] Test feature parity with old system

## Success Metrics

### Functional
- [ ] All existing features work as agents
- [ ] LLM can dynamically choose which agents to use
- [ ] Agent combinations work correctly
- [ ] Performance is equivalent or better than current system

### Technical
- [ ] Code complexity reduced
- [ ] Feature coupling eliminated
- [ ] Extensibility improved
- [ ] Test coverage maintained or improved

### User Experience
- [ ] No regression in functionality
- [ ] Improved response relevance through dynamic agent selection
- [ ] Better composability of features
- [ ] Easier configuration and customization

## Risks and Mitigations

### Risk: Performance Degradation
**Mitigation**: Implement caching, optimize agent execution, performance testing

### Risk: LLM Tool Selection Quality
**Mitigation**: Improve tool descriptions, add examples, implement fallback logic

### Risk: Breaking Changes
**Mitigation**: Phased migration, feature flags, comprehensive testing

### Risk: Increased Complexity
**Mitigation**: Clear interfaces, good documentation, gradual rollout

## Conclusion

This refactor will transform Open WebUI from a rigid feature pipeline to a flexible, LLM-driven agent system. The migration strategy ensures minimal disruption while enabling powerful new capabilities for dynamic feature composition and extensibility.