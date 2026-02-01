"""PocketPaw Web Dashboard - API Server

Lightweight FastAPI server that serves the frontend and handles WebSocket communication.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pocketclaw.config import Settings
from pocketclaw.llm.router import LLMRouter
from pocketclaw.agents.router import AgentRouter
from pocketclaw.scheduler import get_scheduler
from pocketclaw.daemon import get_daemon
from pocketclaw.skills import get_skill_loader, SkillExecutor

logger = logging.getLogger(__name__)

# Active WebSocket connections for reminder notifications
active_connections: list[WebSocket] = []

# Get frontend directory
FRONTEND_DIR = Path(__file__).parent / "frontend"

# Create FastAPI app
app = FastAPI(title="PocketPaw Dashboard")

# Allow CORS for WebSocket
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


async def broadcast_reminder(reminder: dict):
    """Broadcast a reminder notification to all connected clients."""
    message = {
        "type": "reminder",
        "reminder": reminder
    }
    for ws in active_connections[:]:  # Copy list to avoid modification during iteration
        try:
            await ws.send_json(message)
        except Exception:
            # Connection might be closed
            if ws in active_connections:
                active_connections.remove(ws)


async def broadcast_intention(intention_id: str, chunk: dict):
    """Broadcast intention execution results to all connected clients."""
    message = {
        "type": "intention_event",
        "intention_id": intention_id,
        **chunk
    }
    for ws in active_connections[:]:
        try:
            await ws.send_json(message)
        except Exception:
            if ws in active_connections:
                active_connections.remove(ws)


@app.on_event("startup")
async def startup_event():
    """Start the scheduler and daemon on app startup."""
    # Start reminder scheduler
    scheduler = get_scheduler()
    scheduler.start(callback=broadcast_reminder)
    logger.info("Reminder scheduler started")

    # Start proactive daemon
    daemon = get_daemon()
    daemon.start(stream_callback=broadcast_intention)
    logger.info("Proactive daemon started")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the scheduler and daemon on app shutdown."""
    # Stop proactive daemon
    daemon = get_daemon()
    daemon.stop()
    logger.info("Proactive daemon stopped")

    # Stop reminder scheduler
    scheduler = get_scheduler()
    scheduler.stop()
    logger.info("Reminder scheduler stopped")


@app.get("/")
async def index():
    """Serve the main dashboard page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()

    # Track connection for reminder broadcasts
    active_connections.append(websocket)

    # Send welcome notification (not chat message)
    await websocket.send_json({
        "type": "notification",
        "content": "👋 Connected to PocketPaw!"
    })

    # Load settings
    settings = Settings()
    
    # State
    agent_active = False
    llm_router: Optional[LLMRouter] = None
    agent_router: Optional[AgentRouter] = None
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            # Handle tool requests
            if action == "tool":
                tool = data.get("tool")
                await handle_tool(websocket, tool, settings, data)
            
            # Handle agent toggle
            elif action == "toggle_agent":
                agent_active = data.get("active", False)
                
                if agent_active and not agent_router:
                    agent_router = AgentRouter(settings)
                
                await websocket.send_json({
                    "type": "notification",
                    "content": f"🧠 Agent Mode: {'ON' if agent_active else 'OFF'}"
                })
            
            # Handle chat
            elif action == "chat":
                message = data.get("message", "")
                
                if agent_active and agent_router:
                    # Stream agent responses
                    await websocket.send_json({"type": "stream_start"})
                    try:
                        async for chunk in agent_router.run(message):
                            await websocket.send_json(chunk)
                    finally:
                        await websocket.send_json({"type": "stream_end"})
                else:
                    # Simple LLM response
                    if not llm_router:
                        llm_router = LLMRouter(settings)
                    
                    await websocket.send_json({"type": "stream_start"})
                    try:
                        response = await llm_router.chat(message)
                        await websocket.send_json({
                            "type": "message",
                            "content": response
                        })
                    finally:
                        await websocket.send_json({"type": "stream_end"})
            
            # Handle settings update
            elif action == "settings":
                settings.agent_backend = data.get("agent_backend", settings.agent_backend)
                settings.llm_provider = data.get("llm_provider", settings.llm_provider)
                settings.save()
                
                # Reset routers to pick up new settings
                llm_router = None
                agent_router = None
                
                await websocket.send_json({
                    "type": "message",
                    "content": "⚙️ Settings updated"
                })
            
            # Handle API key save
            elif action == "save_api_key":
                provider = data.get("provider")
                key = data.get("key", "")
                
                if provider == "anthropic" and key:
                    settings.anthropic_api_key = key
                    settings.llm_provider = "anthropic"
                    settings.save()
                    llm_router = None
                    agent_router = None
                    await websocket.send_json({
                        "type": "message",
                        "content": "✅ Anthropic API key saved!"
                    })
                elif provider == "openai" and key:
                    settings.openai_api_key = key
                    settings.llm_provider = "openai"
                    settings.save()
                    llm_router = None
                    agent_router = None
                    await websocket.send_json({
                        "type": "message",
                        "content": "✅ OpenAI API key saved!"
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Invalid API key or provider"
                    })
            
            # Handle file navigation (legacy)
            elif action == "navigate":
                path = data.get("path", "")
                await handle_file_navigation(websocket, path, settings)

            # Handle file browser
            elif action == "browse":
                path = data.get("path", "~")
                await handle_file_browse(websocket, path, settings)

            # Handle reminder actions
            elif action == "get_reminders":
                scheduler = get_scheduler()
                reminders = scheduler.get_reminders()
                # Add time remaining to each reminder
                for r in reminders:
                    r["time_remaining"] = scheduler.format_time_remaining(r)
                await websocket.send_json({
                    "type": "reminders",
                    "reminders": reminders
                })

            elif action == "add_reminder":
                message = data.get("message", "")
                scheduler = get_scheduler()
                reminder = scheduler.add_reminder(message)

                if reminder:
                    reminder["time_remaining"] = scheduler.format_time_remaining(reminder)
                    await websocket.send_json({
                        "type": "reminder_added",
                        "reminder": reminder
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Could not parse time from message. Try 'in 5 minutes' or 'at 3pm'"
                    })

            elif action == "delete_reminder":
                reminder_id = data.get("id", "")
                scheduler = get_scheduler()
                if scheduler.delete_reminder(reminder_id):
                    await websocket.send_json({
                        "type": "reminder_deleted",
                        "id": reminder_id
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Reminder not found"
                    })

            # ==================== Intentions API ====================

            elif action == "get_intentions":
                daemon = get_daemon()
                intentions = daemon.get_intentions()
                await websocket.send_json({
                    "type": "intentions",
                    "intentions": intentions
                })

            elif action == "create_intention":
                daemon = get_daemon()
                try:
                    intention = daemon.create_intention(
                        name=data.get("name", "Unnamed"),
                        prompt=data.get("prompt", ""),
                        trigger=data.get("trigger", {"type": "cron", "schedule": "0 9 * * *"}),
                        context_sources=data.get("context_sources", []),
                        enabled=data.get("enabled", True)
                    )
                    await websocket.send_json({
                        "type": "intention_created",
                        "intention": intention
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Failed to create intention: {e}"
                    })

            elif action == "update_intention":
                daemon = get_daemon()
                intention_id = data.get("id", "")
                updates = data.get("updates", {})
                intention = daemon.update_intention(intention_id, updates)
                if intention:
                    await websocket.send_json({
                        "type": "intention_updated",
                        "intention": intention
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Intention not found"
                    })

            elif action == "delete_intention":
                daemon = get_daemon()
                intention_id = data.get("id", "")
                if daemon.delete_intention(intention_id):
                    await websocket.send_json({
                        "type": "intention_deleted",
                        "id": intention_id
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Intention not found"
                    })

            elif action == "toggle_intention":
                daemon = get_daemon()
                intention_id = data.get("id", "")
                intention = daemon.toggle_intention(intention_id)
                if intention:
                    await websocket.send_json({
                        "type": "intention_toggled",
                        "intention": intention
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Intention not found"
                    })

            elif action == "run_intention":
                daemon = get_daemon()
                intention_id = data.get("id", "")
                intention = daemon.get_intention(intention_id)
                if intention:
                    # Run in background, results streamed via broadcast_intention
                    await websocket.send_json({
                        "type": "notification",
                        "content": f"🚀 Running intention: {intention['name']}"
                    })
                    asyncio.create_task(daemon.run_intention_now(intention_id))
                else:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Intention not found"
                    })

            # ==================== Skills API ====================

            elif action == "get_skills":
                loader = get_skill_loader()
                loader.reload()  # Refresh to catch new installs
                skills = [
                    {
                        "name": s.name,
                        "description": s.description,
                        "argument_hint": s.argument_hint,
                    }
                    for s in loader.get_invocable()
                ]
                await websocket.send_json({
                    "type": "skills",
                    "skills": skills
                })

            elif action == "run_skill":
                skill_name = data.get("name", "")
                skill_args = data.get("args", "")

                loader = get_skill_loader()
                skill = loader.get(skill_name)

                if not skill:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Skill not found: {skill_name}"
                    })
                else:
                    await websocket.send_json({
                        "type": "notification",
                        "content": f"🎯 Running skill: {skill_name}"
                    })

                    # Execute skill through agent
                    executor = SkillExecutor(settings)
                    await websocket.send_json({"type": "stream_start"})
                    try:
                        async for chunk in executor.execute_skill(skill, skill_args):
                            await websocket.send_json(chunk)
                    finally:
                        await websocket.send_json({"type": "stream_end"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Remove connection from tracking
        if websocket in active_connections:
            active_connections.remove(websocket)


async def handle_tool(websocket: WebSocket, tool: str, settings: Settings, data: dict):
    """Handle tool execution."""
    
    if tool == "status":
        from pocketclaw.tools.status import get_system_status
        status = get_system_status()  # sync function
        await websocket.send_json({
            "type": "status",
            "content": status
        })
    
    elif tool == "screenshot":
        from pocketclaw.tools.screenshot import take_screenshot
        result = take_screenshot()  # sync function
        
        if isinstance(result, bytes):
            await websocket.send_json({
                "type": "screenshot",
                "image": base64.b64encode(result).decode()
            })
        else:
            await websocket.send_json({
                "type": "error",
                "content": result
            })
    
    elif tool == "fetch":
        from pocketclaw.tools.fetch import list_directory
        path = data.get("path") or str(Path.home())
        result = list_directory(path, settings.file_jail_path)  # sync function
        await websocket.send_json({
            "type": "message",
            "content": result
        })
    
    elif tool == "panic":
        await websocket.send_json({
            "type": "message",
            "content": "🛑 PANIC: All agent processes stopped!"
        })
        # TODO: Actually stop agent processes
    
    else:
        await websocket.send_json({
            "type": "error",
            "content": f"Unknown tool: {tool}"
        })


async def handle_file_navigation(websocket: WebSocket, path: str, settings: Settings):
    """Handle file browser navigation."""
    from pocketclaw.tools.fetch import list_directory

    result = list_directory(path, settings.file_jail_path)  # sync function
    await websocket.send_json({
        "type": "message",
        "content": result
    })


async def handle_file_browse(websocket: WebSocket, path: str, settings: Settings):
    """Handle file browser - returns structured JSON for the modal."""
    from pocketclaw.tools.fetch import is_safe_path

    # Resolve ~ to home directory
    if path == "~" or path == "":
        resolved_path = Path.home()
    else:
        # Handle relative paths from home
        if not path.startswith("/"):
            resolved_path = Path.home() / path
        else:
            resolved_path = Path(path)

    resolved_path = resolved_path.resolve()
    jail = settings.file_jail_path.resolve()

    # Security check
    if not is_safe_path(resolved_path, jail):
        await websocket.send_json({
            "type": "files",
            "error": "Access denied: path outside allowed directory"
        })
        return

    if not resolved_path.exists():
        await websocket.send_json({
            "type": "files",
            "error": "Path does not exist"
        })
        return

    if not resolved_path.is_dir():
        await websocket.send_json({
            "type": "files",
            "error": "Not a directory"
        })
        return

    # Build file list
    files = []
    try:
        items = sorted(resolved_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

        for item in items[:50]:  # Limit to 50 items
            if item.name.startswith("."):
                continue  # Skip hidden files

            file_info = {
                "name": item.name,
                "isDir": item.is_dir()
            }

            if not item.is_dir():
                try:
                    size = item.stat().st_size
                    if size < 1024:
                        file_info["size"] = f"{size} B"
                    elif size < 1024 * 1024:
                        file_info["size"] = f"{size/1024:.1f} KB"
                    else:
                        file_info["size"] = f"{size/(1024*1024):.1f} MB"
                except Exception:
                    file_info["size"] = "?"

            files.append(file_info)

    except PermissionError:
        await websocket.send_json({
            "type": "files",
            "error": "Permission denied"
        })
        return

    # Calculate relative path from home for display
    try:
        rel_path = resolved_path.relative_to(Path.home())
        display_path = str(rel_path) if str(rel_path) != "." else "~"
    except ValueError:
        display_path = str(resolved_path)

    await websocket.send_json({
        "type": "files",
        "path": display_path,
        "files": files
    })


def run_dashboard(host: str = "127.0.0.1", port: int = 8888):
    """Run the dashboard server."""
    print("\n" + "=" * 50)
    print("🐾 POCKETPAW WEB DASHBOARD")
    print("=" * 50)
    print(f"\n🌐 Open http://localhost:{port} in your browser\n")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()
