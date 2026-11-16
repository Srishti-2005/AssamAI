from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.db_models import Agent
from app.models.schemas import AgentRegisterRequest, AgentResponse
from app.services.embedding import embedding_service
import json

router = APIRouter(prefix="/agents", tags=["Agents"])

@router.post("/register", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def register_agent(req: AgentRegisterRequest, db: Session = Depends(get_db)):
    """
    Registers a new enterprise AI agent, generates a secure HMAC webhook secret,
    and initializes its semantic interest vector.
    """
    # Check for duplicate registration by company name
    existing = db.query(Agent).filter(Agent.company_name == req.company_name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with company name '{req.company_name}' is already registered."
        )

    # Compute initial interest embedding from company manifesto
    text_to_embed = f"{req.company_name} {req.industry_category} {req.interest_manifesto}"
    embedding = embedding_service.embed_text(text_to_embed)

    agent = Agent(
        company_name=req.company_name,
        industry_category=req.industry_category,
        webhook_url=req.webhook_url,
        interest_manifesto=req.interest_manifesto,
        blocked_industries=json.dumps(req.blocked_industries or [])
    )
    agent.set_embedding(embedding)

    db.add(agent)
    db.commit()
    db.refresh(agent)

    return agent

@router.get("", response_model=List[AgentResponse])
def list_agents(is_active: Optional[bool] = None, db: Session = Depends(get_db)):
    """
    Lists all registered enterprise agents.
    """
    query = db.query(Agent)
    if is_active is not None:
        query = query.filter(Agent.is_active == is_active)
    return query.all()

@router.get("/{agent_id}", response_model=AgentResponse)
def get_agent(agent_id: str, db: Session = Depends(get_db)):
    """
    Retrieves agent profile details.
    """
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
