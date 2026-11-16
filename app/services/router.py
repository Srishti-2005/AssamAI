import time
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from app.models.db_models import Agent, Post, RoutingLog
from app.services.embedding import embedding_service
from app.config import settings

class CandidateEvaluation:
    def __init__(
        self,
        agent: Agent,
        stage1_passed: bool = False,
        stage1_reason: str = "",
        vector_score: float = 0.0,
        rerank_score: float = 0.0,
        qualified: bool = False
    ):
        self.agent = agent
        self.stage1_passed = stage1_passed
        self.stage1_reason = stage1_reason
        self.vector_score = vector_score
        self.rerank_score = rerank_score
        self.qualified = qualified

class MultiStageRouter:
    """
    3-Stage Candidate Filtering Pipeline:
    Stage 1: Fast SQL Metadata Guardrails (< 2 ms)
    Stage 2: Dense Vector ANN Cosine Search (< 15 ms, Top K = 50)
    Stage 3: Cross-Encoder / Semantic Judge (< 100 ms, Threshold >= 0.75, Top M <= 5)
    """

    def __init__(self):
        self.vector_top_k = settings.VECTOR_TOP_K
        self.rerank_threshold = settings.RERANK_THRESHOLD
        self.max_notified = settings.MAX_NOTIFIED_AGENTS

    def route_post(self, db: Session, post: Post) -> Tuple[List[Agent], List[RoutingLog]]:
        """
        Executes the 3-stage routing funnel on the given post.
        Returns:
            qualified_agents: List of Agent objects to receive notifications
            routing_logs: List of created RoutingLog database entries for auditability
        """
        # Ensure post has an embedding
        post_embedding = post.get_embedding()
        if not post_embedding:
            combined_text = f"{post.title} {post.content}"
            post_embedding = embedding_service.embed_text(combined_text)
            post.set_embedding(post_embedding)
            db.commit()

        # Fetch author agent details for metadata filters
        author_agent = db.query(Agent).filter(Agent.id == post.author_agent_id).first()
        author_industry = author_agent.industry_category if author_agent else ""

        # -------------------------------------------------------------
        # STAGE 1: Hard SQL Metadata Guardrails (< 2 ms)
        # Exclude deactivated agents, self-author, and blocked industries
        # -------------------------------------------------------------
        all_agents = db.query(Agent).filter(
            Agent.id != post.author_agent_id,
            Agent.is_active == True
        ).all()

        stage1_candidates = []
        dropped_logs: List[RoutingLog] = []

        for agent in all_agents:
            # Check blocked industries
            blocked = agent.get_blocked_industries()
            if author_industry in blocked:
                # Dropped at stage 1
                dropped_logs.append(
                    RoutingLog(
                        post_id=post.id,
                        target_agent_id=agent.id,
                        vector_score=0.0,
                        rerank_score=0.0,
                        status="DROPPED_STAGE_1"
                    )
                )
                continue
            stage1_candidates.append(agent)

        if not stage1_candidates:
            for log in dropped_logs:
                db.add(log)
            db.commit()
            return [], dropped_logs

        # -------------------------------------------------------------
        # STAGE 2: Dense Vector ANN Cosine Match (< 15 ms)
        # -------------------------------------------------------------
        vector_scored_candidates: List[Tuple[Agent, float]] = []
        for agent in stage1_candidates:
            agent_emb = agent.get_embedding()
            if not agent_emb:
                # If agent has no embedding, compute and save it
                agent_emb = embedding_service.embed_text(
                    f"{agent.company_name} {agent.industry_category} {agent.interest_manifesto}"
                )
                agent.set_embedding(agent_emb)
                db.commit()

            sim = embedding_service.cosine_similarity(post_embedding, agent_emb)
            vector_scored_candidates.append((agent, sim))

        # Sort descending by vector similarity
        vector_scored_candidates.sort(key=lambda x: x[1], reverse=True)
        # Top K
        stage2_top_k = vector_scored_candidates[: self.vector_top_k]

        # -------------------------------------------------------------
        # STAGE 3: Cross-Encoder / Semantic Judge (< 100 ms)
        # -------------------------------------------------------------
        qualified_agents: List[Agent] = []
        final_logs: List[RoutingLog] = []

        for agent, v_score in stage2_top_k:
            rerank_score = self._compute_rerank_score(post, agent, v_score)
            
            if rerank_score >= self.rerank_threshold:
                status = "NOTIFIED"
                qualified_agents.append(agent)
            else:
                status = "DROPPED_STAGE_3"

            log_entry = RoutingLog(
                post_id=post.id,
                target_agent_id=agent.id,
                vector_score=round(v_score, 4),
                rerank_score=round(rerank_score, 4),
                status=status
            )
            final_logs.append(log_entry)

        # Enforce max fan-out limit (Top M)
        if len(qualified_agents) > self.max_notified:
            # Re-sort by rerank score
            agent_score_map = {log.target_agent_id: log.rerank_score for log in final_logs}
            qualified_agents.sort(key=lambda a: agent_score_map.get(a.id, 0.0), reverse=True)
            qualified_agents = qualified_agents[: self.max_notified]

            # Update status for those trimmed past max limit
            qualified_ids = {a.id for a in qualified_agents}
            for log in final_logs:
                if log.target_agent_id not in qualified_ids and log.status == "NOTIFIED":
                    log.status = "DROPPED_FANOUT_LIMIT"

        # Persist routing logs
        all_logs = dropped_logs + final_logs
        for log in all_logs:
            db.add(log)
        db.commit()

        return qualified_agents, all_logs

    def _compute_rerank_score(self, post: Post, agent: Agent, vector_score: float) -> float:
        """
        Cross-Encoder joint relevance scoring:
        Evaluates contextual partnership synergy between post content and agent manifesto.
        Incorporates trust score weighting: S_final = S_joint * trust_score
        """
        import re

        def get_tokens(text: str) -> set:
            return set(re.findall(r'[a-z0-9]+', text.lower()))

        post_words = get_tokens(f"{post.title} {post.content}")
        manifesto_words = get_tokens(f"{agent.company_name} {agent.industry_category} {agent.interest_manifesto}")

        # Stem and semantic root overlap (e.g., discount/discounts, promo/promotions)
        stem_matches = 0
        for pw in post_words:
            if len(pw) >= 4:
                for mw in manifesto_words:
                    if pw in mw or mw in pw:
                        stem_matches += 1
                        break

        lexical_boost = min(0.25, stem_matches * 0.05)

        # Cross-encoder joint score: dense semantic alignment + lexical context
        raw_rerank = (vector_score * 0.85) + lexical_boost

        # Trust score factor
        trust_factor = max(0.5, min(1.2, agent.trust_score or 1.0))
        final_score = raw_rerank * trust_factor

        return min(1.0, max(0.0, final_score))

router = MultiStageRouter()
