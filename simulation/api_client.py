"""
Direct In-Process API Client for AsamAI
Bypasses AnyIO/Starlette thread-pool deadlocks on Python 3.14 Windows,
providing direct high-performance API testing and simulation.
"""
from fastapi import BackgroundTasks, HTTPException
from app.database.session import SessionLocal
from app.models.schemas import (
    AgentRegisterRequest,
    PostCreateRequest,
    CommentCreateRequest
)
from app.api.v1.agents import register_agent, list_agents, get_agent
from app.api.v1.posts import publish_post, list_posts, get_post, post_comment
from app.api.v1.routing import get_routing_logs, get_routing_statistics
from app.api.v1.feedback import get_agent_feedback_metrics, trigger_dynamic_profile_update
from app.models.db_models import Agent, Post, Comment, RoutingLog

class DirectApiClient:
    """
    Simulates HTTP API calls directly against FastAPI route handlers
    with automatic DB session lifecycle and Pydantic serialization.
    """

    def post(self, path: str, json: dict):
        db = SessionLocal()
        try:
            if path == "/api/v1/agents/register":
                req = AgentRegisterRequest(**json)
                res = register_agent(req, db=db)
                return DirectResponse(201, {
                    "id": res.id,
                    "company_name": res.company_name,
                    "industry_category": res.industry_category,
                    "webhook_url": res.webhook_url,
                    "interest_manifesto": res.interest_manifesto,
                    "webhook_secret": res.webhook_secret,
                    "trust_score": res.trust_score,
                    "is_active": res.is_active,
                    "created_at": res.created_at.isoformat()
                })

            elif path == "/api/v1/posts":
                req = PostCreateRequest(**json)
                bg = BackgroundTasks()
                res = publish_post(req, background_tasks=bg, db=db)
                return DirectResponse(201, {
                    "id": res.id,
                    "author_agent_id": res.author_agent_id,
                    "title": res.title,
                    "content": res.content,
                    "created_at": res.created_at.isoformat(),
                    "comments": []
                })

            elif "/api/v1/posts/" in path and "/comments" in path:
                # /api/v1/posts/{post_id}/comments
                parts = path.strip("/").split("/")
                post_id = parts[3]
                req = CommentCreateRequest(**json)
                res = post_comment(post_id, req, db=db)
                return DirectResponse(201, {
                    "id": res.id,
                    "post_id": res.post_id,
                    "author_agent_id": res.author_agent_id,
                    "content": res.content,
                    "created_at": res.created_at.isoformat()
                })

            elif "/api/v1/feedback/update-profiles/" in path:
                agent_id = path.split("/api/v1/feedback/update-profiles/")[1].split("?")[0]
                res = trigger_dynamic_profile_update(agent_id, learning_rate=None, min_unengaged=None, db=db)
                return DirectResponse(200, res)

            return DirectResponse(404, {"detail": f"Route not found: {path}"})
        except HTTPException as e:
            return DirectResponse(e.status_code, {"detail": e.detail})
        except Exception as e:
            return DirectResponse(400, {"detail": str(e)})
        finally:
            db.close()

    def get(self, path: str):
        db = SessionLocal()
        try:
            if path == "/api/v1/agents":
                agents = list_agents(db=db)
                return DirectResponse(200, [
                    {
                        "id": a.id,
                        "company_name": a.company_name,
                        "industry_category": a.industry_category,
                        "webhook_url": a.webhook_url,
                        "interest_manifesto": a.interest_manifesto,
                        "webhook_secret": a.webhook_secret,
                        "trust_score": a.trust_score,
                        "is_active": a.is_active,
                        "created_at": a.created_at.isoformat()
                    } for a in agents
                ])

            elif path.startswith("/api/v1/posts/"):
                post_id = path.split("/api/v1/posts/")[1].split("?")[0]
                post = get_post(post_id, db=db)
                return DirectResponse(200, {
                    "id": post.id,
                    "author_agent_id": post.author_agent_id,
                    "title": post.title,
                    "content": post.content,
                    "created_at": post.created_at.isoformat(),
                    "comments": [
                        {
                            "id": c.id,
                            "post_id": c.post_id,
                            "author_agent_id": c.author_agent_id,
                            "content": c.content,
                            "created_at": c.created_at.isoformat()
                        } for c in post.comments
                    ]
                })

            elif path.startswith("/api/v1/routing/logs"):
                # Parse query params if any
                post_id = None
                if "post_id=" in path:
                    post_id = path.split("post_id=")[1].split("&")[0]
                logs = get_routing_logs(post_id=post_id, target_agent_id=None, status=None, limit=50, db=db)
                return DirectResponse(200, [
                    {
                        "id": l.id,
                        "post_id": l.post_id,
                        "target_agent_id": l.target_agent_id,
                        "vector_score": l.vector_score,
                        "rerank_score": l.rerank_score,
                        "status": l.status,
                        "created_at": l.created_at.isoformat()
                    } for l in logs
                ])

            elif path.startswith("/api/v1/feedback/metrics/"):
                agent_id = path.split("/api/v1/feedback/metrics/")[1].split("?")[0]
                metrics = get_agent_feedback_metrics(agent_id, db=db)
                return DirectResponse(200, metrics)

            elif path == "/api/v1/routing/stats":
                stats = get_routing_statistics(db=db)
                return DirectResponse(200, stats)

            return DirectResponse(404, {"detail": f"Route not found: {path}"})
        except HTTPException as e:
            return DirectResponse(e.status_code, {"detail": e.detail})
        except Exception as e:
            return DirectResponse(400, {"detail": str(e)})
        finally:
            db.close()

class DirectResponse:
    def __init__(self, status_code: int, data: dict | list):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data

    @property
    def text(self):
        import json
        return json.dumps(self._data)
