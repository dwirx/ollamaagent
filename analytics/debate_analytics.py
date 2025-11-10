from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx
import pandas as pd
from langfuse.openai import OpenAI

from council.types import DebateState, Argument, Vote


@dataclass
class AgentStats:
    """Statistics for a single agent"""
    name: str
    total_arguments: int
    avg_focus_score: float
    first_place_votes: int
    total_votes_received: int
    win_rate: float
    avg_argument_length: float
    participations: int


@dataclass
class DebateAnalytics:
    """Comprehensive analytics for a debate"""
    debate_id: str
    question: str
    total_iterations: int
    consensus_reached: bool
    winner: Optional[str]

    # Agent performance
    agent_stats: Dict[str, AgentStats]

    # Voting patterns
    voting_matrix: Dict[str, Dict[str, int]]  # voter -> {voted_for -> count}

    # Argument network
    argument_graph_data: Dict[str, Any]

    # Sentiment analysis
    sentiment_scores: Dict[str, float]  # agent -> avg sentiment

    # Temporal data
    iteration_consensus_scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "debate_id": self.debate_id,
            "question": self.question,
            "total_iterations": self.total_iterations,
            "consensus_reached": self.consensus_reached,
            "winner": self.winner,
            "agent_stats": {k: asdict(v) for k, v in self.agent_stats.items()},
            "voting_matrix": self.voting_matrix,
            "argument_graph": self.argument_graph_data,
            "sentiment_scores": self.sentiment_scores,
            "iteration_consensus_scores": self.iteration_consensus_scores,
        }


class DebateAnalyzer:
    """Analyzes debates and generates insights"""

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client
        self.debate_history: List[DebateState] = []

    def analyze_debate(self, state: DebateState) -> DebateAnalytics:
        """
        Comprehensive analysis of a single debate

        Args:
            state: DebateState to analyze

        Returns:
            DebateAnalytics with complete insights
        """
        # Extract basic info
        debate_id = f"debate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        question = state.config.question
        total_iterations = len(state.iterations)
        consensus_reached = state.iterations[-1].consensus_reached if state.iterations else False
        winner = state.iterations[-1].consensus_candidate if state.iterations else None

        # Analyze agents
        agent_stats = self._calculate_agent_stats(state)

        # Analyze voting patterns
        voting_matrix = self._build_voting_matrix(state)

        # Build argument graph
        argument_graph = self._build_argument_graph(state)

        # Sentiment analysis (if client available)
        sentiment_scores = self._analyze_sentiments(state) if self.client else {}

        # Consensus progression
        consensus_progression = self._calculate_consensus_progression(state)

        return DebateAnalytics(
            debate_id=debate_id,
            question=question,
            total_iterations=total_iterations,
            consensus_reached=consensus_reached,
            winner=winner,
            agent_stats=agent_stats,
            voting_matrix=voting_matrix,
            argument_graph_data=argument_graph,
            sentiment_scores=sentiment_scores,
            iteration_consensus_scores=consensus_progression,
        )

    def _calculate_agent_stats(self, state: DebateState) -> Dict[str, AgentStats]:
        """Calculate performance statistics for each agent"""
        stats = {}

        # Collect all arguments per agent
        agent_arguments: Dict[str, List[Argument]] = defaultdict(list)
        for iteration in state.iterations:
            for arg in iteration.arguments:
                agent_arguments[arg.author].append(arg)

        # Collect voting data
        first_place_counts: Dict[str, int] = defaultdict(int)
        total_vote_counts: Dict[str, int] = defaultdict(int)

        for iteration in state.iterations:
            for vote in iteration.votes:
                if vote.ranking:
                    # First place
                    first_place_counts[vote.ranking[0]] += 1

                    # All votes
                    for name in vote.ranking:
                        total_vote_counts[name] += 1

        # Calculate win rate (# of first place / total participations)
        participations = len(state.iterations)

        for agent_name in agent_arguments.keys():
            arguments = agent_arguments[agent_name]
            first_place = first_place_counts.get(agent_name, 0)
            win_rate = first_place / participations if participations > 0 else 0.0

            avg_length = sum(len(arg.content) for arg in arguments) / len(arguments) if arguments else 0.0

            stats[agent_name] = AgentStats(
                name=agent_name,
                total_arguments=len(arguments),
                avg_focus_score=0.8,  # TODO: integrate with focus_scorer
                first_place_votes=first_place,
                total_votes_received=total_vote_counts.get(agent_name, 0),
                win_rate=win_rate,
                avg_argument_length=avg_length,
                participations=participations,
            )

        return stats

    def _build_voting_matrix(self, state: DebateState) -> Dict[str, Dict[str, int]]:
        """Build matrix of who voted for whom"""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for iteration in state.iterations:
            for vote in iteration.votes:
                voter = vote.voter
                if vote.ranking:
                    # First choice gets highest weight
                    for idx, voted_for in enumerate(vote.ranking):
                        weight = len(vote.ranking) - idx
                        matrix[voter][voted_for] += weight

        return {k: dict(v) for k, v in matrix.items()}

    def _build_argument_graph(self, state: DebateState) -> Dict[str, Any]:
        """
        Build a graph showing relationships between arguments
        Returns NetworkX-compatible JSON
        """
        G = nx.DiGraph()

        # Add nodes for each argument
        arg_counter = 0
        arg_id_map = {}

        for iteration in state.iterations:
            for arg in iteration.arguments:
                arg_id = f"arg_{arg_counter}"
                arg_id_map[arg_counter] = arg_id
                G.add_node(
                    arg_id,
                    author=arg.author,
                    iteration=arg.iteration,
                    content=arg.content[:100] + "..." if len(arg.content) > 100 else arg.content,
                )
                arg_counter += 1

        # Add edges based on iteration sequence (arguments in iteration N respond to iteration N-1)
        prev_iteration_args = []
        for iteration in state.iterations:
            current_args = []
            for arg in iteration.arguments:
                arg_id = [k for k, v in arg_id_map.items() if G.nodes[v]['author'] == arg.author and G.nodes[v]['iteration'] == arg.iteration]
                if arg_id:
                    current_arg_id = arg_id_map[arg_id[0]]
                    current_args.append(current_arg_id)

                    # Connect to previous iteration arguments
                    for prev_arg_id in prev_iteration_args:
                        G.add_edge(prev_arg_id, current_arg_id, weight=0.5)

            prev_iteration_args = current_args

        # Convert to JSON-serializable format
        return nx.node_link_data(G)

    def _analyze_sentiments(self, state: DebateState) -> Dict[str, float]:
        """
        Analyze sentiment/tone of arguments per agent

        Returns sentiment score: -1.0 (negative) to 1.0 (positive)
        """
        if not self.client:
            return {}

        agent_sentiments: Dict[str, List[float]] = defaultdict(list)

        for iteration in state.iterations:
            for arg in iteration.arguments:
                # Simple sentiment analysis via LLM
                try:
                    prompt = (
                        f"Analisis sentimen/tone dari argumen berikut. "
                        f"Berikan score -1.0 (sangat negatif/agresif) hingga 1.0 (sangat positif/konstruktif).\n\n"
                        f"Argumen: {arg.content[:500]}\n\n"
                        f"Jawab HANYA dengan angka, contoh: 0.75"
                    )

                    response = self.client.chat.completions.create(
                        model="gemma3:1b",
                        messages=[
                            {"role": "system", "content": "You are a sentiment analyzer. Reply only with a number between -1.0 and 1.0."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                    )

                    score_text = response.choices[0].message.content.strip()
                    score = float(score_text)
                    score = max(-1.0, min(1.0, score))  # Clamp

                    agent_sentiments[arg.author].append(score)

                except Exception as e:
                    print(f"Sentiment analysis error: {e}")
                    agent_sentiments[arg.author].append(0.0)

        # Average per agent
        return {
            agent: sum(scores) / len(scores) if scores else 0.0
            for agent, scores in agent_sentiments.items()
        }

    def _calculate_consensus_progression(self, state: DebateState) -> List[float]:
        """Calculate how consensus evolved over iterations"""
        scores = []

        for iteration in state.iterations:
            if not iteration.votes:
                scores.append(0.0)
                continue

            # Calculate consensus as concentration of first-place votes
            first_place_counts: Dict[str, int] = defaultdict(int)
            total_votes = len(iteration.votes)

            for vote in iteration.votes:
                if vote.ranking:
                    first_place_counts[vote.ranking[0]] += 1

            # Maximum first place percentage = consensus score
            max_first_place = max(first_place_counts.values()) if first_place_counts else 0
            consensus_score = max_first_place / total_votes if total_votes > 0 else 0.0

            scores.append(consensus_score)

        return scores

    def load_debate_from_file(self, file_path: Path) -> DebateState:
        """Load debate state from JSON file"""
        with open(file_path) as f:
            data = json.load(f)

        # TODO: Implement proper deserialization
        # For now, return as-is
        from council.types import DebateState
        from pydantic import parse_obj_as

        return parse_obj_as(DebateState, data)

    def aggregate_stats_across_debates(self, debate_files: List[Path]) -> Dict[str, Any]:
        """
        Aggregate statistics across multiple debates

        Returns:
            Aggregated stats including win rates, participation rates, etc.
        """
        all_agent_stats: Dict[str, List[AgentStats]] = defaultdict(list)

        for file_path in debate_files:
            try:
                state = self.load_debate_from_file(file_path)
                analytics = self.analyze_debate(state)

                for agent_name, stats in analytics.agent_stats.items():
                    all_agent_stats[agent_name].append(stats)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        # Aggregate
        aggregated = {}
        for agent_name, stats_list in all_agent_stats.items():
            aggregated[agent_name] = {
                "total_debates": len(stats_list),
                "total_arguments": sum(s.total_arguments for s in stats_list),
                "avg_win_rate": sum(s.win_rate for s in stats_list) / len(stats_list),
                "total_first_place": sum(s.first_place_votes for s in stats_list),
                "avg_focus_score": sum(s.avg_focus_score for s in stats_list) / len(stats_list),
            }

        return aggregated
