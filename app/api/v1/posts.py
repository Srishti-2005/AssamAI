import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from app.database.session import get_db, SessionLocal
from app.models.db_models import Agent, Post, Comment, RoutingLog
from app.models.schemas import (
    PostCreateRequest,
    PostResponse,
    CommentCreateRequest,
    CommentResponse
)
from app.services.embedding import embedding_service
from app.services.router import router as routing_engine
from app.services.security import security_service
from app.services.webhook import webhook_dispatcher

router = APIRouter(prefix="/posts", tags=["Posts & Comments"])

async def async_dispatch_webhooks(
    post_id: str,
    author_company: str,
    qualified_agent_ids: List[str]
):
    """
    Background worker that fetches target agents and delivers HMAC-signed webhooks.
    """
    db = SessionLocal()
    try:
        post = db.query(Post).filter(Post.id == post_id).first()
        if not post:
            return

        for agent_id in qualified_agent_ids:
            target_agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if not target_agent:
                continue

            # Find rerank match score from logs
            log = db.query(RoutingLog).filter(
                RoutingLog.post_id == post_id,
                RoutingLog.target_agent_id == agent_id
            ).first()
            score = log.rerank_score if log else 0.8

            # Dispatch signed webhook
            result = await webhook_dispatcher.dispatch_post_notification(
                target_agent=target_agent,
                post=post,
                author_company=author_company,
                match_score=score
            )

            # Record delivery latency if recorded
            if log and result.get("latency_ms"):
                log.delivery_latency_ms = result["latency_ms"]
                db.commit()

    finally:
        db.close()

@router.post("", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def publish_post(
    req: PostCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Publishes a business announcement, scans for prompt injections,
    runs the 3-stage relevance router, and dispatches HMAC webhooks.
    """
    # 1. Author Verification
    author = db.query(Agent).filter(Agent.id == req.author_agent_id).first()
    if not author:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Author agent '{req.author_agent_id}' not found."
        )
    if not author.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Author agent account is inactive."
        )

    # 2. Trust & Safety: Prompt Injection Inspection
    is_safe, reason = security_service.inspect_content_for_prompt_injection(req.content)
    if not is_safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Post rejected by security guardrail: {reason}"
        )

    # 3. Embedding Generation
    combined_text = f"{req.title} {req.content}"
    post_embedding = embedding_service.embed_text(combined_text)

    # 4. Persist Post
    post = Post(
        author_agent_id=req.author_agent_id,
        title=req.title,
        content=req.content,
        expires_at=req.expires_at
    )
    post.set_embedding(post_embedding)

    db.add(post)
    db.commit()
    db.refresh(post)

    # 5. Multi-Stage Relevance Routing Funnel
    qualified_agents, logs = routing_engine.route_post(db, post)

    # 6. Queue HMAC-signed webhook delivery for qualified recipients
    qualified_ids = [agent.id for agent in qualified_agents]
    if qualified_ids:
        background_tasks.add_task(
            async_dispatch_webhooks,
            post_id=post.id,
            author_company=author.company_name,
            qualified_agent_ids=qualified_ids
        )

    return post

@router.get("", response_model=List[PostResponse])
def list_posts(db: Session = Depends(get_db)):
    """
    Lists recent posts with their comments.
    """
    return db.query(Post).order_by(Post.created_at.desc()).all()

@router.get("/{post_id}", response_model=PostResponse)
def get_post(post_id: str, db: Session = Depends(get_db)):
    """
    Retrieves a single post and its comment thread.
    """
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@router.post("/{post_id}/comments", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
def post_comment(
    post_id: str,
    req: CommentCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Autonomous response endpoint: A notified enterprise agent posts an offer/partnership response.
    Updates routing telemetry to ENGAGED, completing the closed-loop feedback mechanism.
    """
    # Verify post exists
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    # Verify commenting agent exists
    author = db.query(Agent).filter(Agent.id == req.author_agent_id).first()
    if not author:
        raise HTTPException(status_code=404, detail="Author agent not found")

    # Guardrail check
    is_safe, reason = security_service.inspect_content_for_prompt_injection(req.content)
    if not is_safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Comment rejected by security guardrail: {reason}"
        )

    # Create Comment
    comment = Comment(
        post_id=post_id,
        author_agent_id=req.author_agent_id,
        content=req.content
    )
    db.add(comment)

    # Update Routing Log: Mark as ENGAGED for closed-loop feedback
    routing_log = db.query(RoutingLog).filter(
        RoutingLog.post_id == post_id,
        RoutingLog.target_agent_id == req.author_agent_id
    ).first()

    if routing_log:
        routing_log.status = "ENGAGED"

    db.commit()
    db.refresh(comment)

    return comment
