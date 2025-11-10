from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

from .types import Personality


class CollaborationStrategy(Enum):
    """Different collaboration strategies"""
    CONSENSUS_BUILDING = "consensus"  # Build toward agreement
    PROBLEM_SOLVING = "problem_solving"  # Solve a complex problem together
    SYNTHESIS = "synthesis"  # Synthesize multiple viewpoints
    BRAINSTORMING = "brainstorming"  # Generate ideas collaboratively


@dataclass
class SubGroup:
    """A sub-group of agents working together"""
    name: str
    members: List[Personality]
    focus: str  # What this sub-group focuses on
    coordinator: Optional[str] = None  # Leader agent name


@dataclass
class Compromise:
    """Track compromises made during collaboration"""
    iteration: int
    agents: List[str]  # Agents involved in compromise
    original_positions: Dict[str, str]  # Agent -> original stance
    compromise_position: str  # The compromised position
    rationale: str  # Why this compromise


@dataclass
class CollaborativeState:
    """State for collaborative debate mode"""
    strategy: CollaborationStrategy
    groups: List[SubGroup]
    shared_workspace: Dict[str, any] = field(default_factory=dict)  # Shared notes/ideas
    compromises: List[Compromise] = field(default_factory=list)
    consensus_items: List[str] = field(default_factory=list)  # Items all agree on
    divergent_items: List[str] = field(default_factory=list)  # Items with disagreement


class CollaborationEngine:
    """
    Engine for collaborative (non-competitive) debates:
    - Agents work together to solve problems
    - Sub-groups can form to tackle different aspects
    - Compromise tracking
    - Consensus building
    """

    def __init__(self, strategy: CollaborationStrategy = CollaborationStrategy.CONSENSUS_BUILDING):
        self.strategy = strategy
        self.state = CollaborativeState(strategy=strategy, groups=[])

    def form_subgroups(
        self,
        personalities: List[Personality],
        num_groups: int = 2,
        strategy: str = "balanced",
    ) -> List[SubGroup]:
        """
        Form sub-groups from list of personalities

        Args:
            personalities: All participating agents
            num_groups: Number of sub-groups to create
            strategy: "balanced" (equal size), "specialized" (by traits)

        Returns:
            List of SubGroup objects
        """
        if strategy == "balanced":
            # Distribute evenly
            groups = []
            group_size = len(personalities) // num_groups

            for i in range(num_groups):
                start = i * group_size
                end = start + group_size if i < num_groups - 1 else len(personalities)
                members = personalities[start:end]

                groups.append(
                    SubGroup(
                        name=f"Team {chr(65+i)}",  # Team A, B, C...
                        members=members,
                        focus=f"Aspect {i+1}",
                        coordinator=members[0].name if members else None,
                    )
                )

            return groups

        elif strategy == "specialized":
            # Group by similar traits/reasoning depth
            # Simple implementation: by reasoning depth
            high_depth = [p for p in personalities if p.reasoning_depth >= 3]
            med_depth = [p for p in personalities if p.reasoning_depth == 2]
            low_depth = [p for p in personalities if p.reasoning_depth == 1]

            groups = []
            if high_depth:
                groups.append(
                    SubGroup(
                        name="Deep Analysis Team",
                        members=high_depth,
                        focus="Complex reasoning and implications",
                        coordinator=high_depth[0].name,
                    )
                )
            if med_depth:
                groups.append(
                    SubGroup(
                        name="Balanced Team",
                        members=med_depth,
                        focus="Practical solutions and trade-offs",
                        coordinator=med_depth[0].name,
                    )
                )
            if low_depth:
                groups.append(
                    SubGroup(
                        name="Quick Response Team",
                        members=low_depth,
                        focus="Rapid assessment and key insights",
                        coordinator=low_depth[0].name,
                    )
                )

            return groups

        return []

    def detect_compromise_opportunity(
        self,
        agent_positions: Dict[str, str],
        iteration: int,
    ) -> Optional[Compromise]:
        """
        Detect if agents' positions can be compromised

        Args:
            agent_positions: Current positions of each agent
            iteration: Current iteration number

        Returns:
            Compromise object if opportunity found
        """
        # Simple heuristic: if 2+ agents have related but not identical positions
        # This is a simplified version; in production, use semantic similarity

        positions_list = list(agent_positions.items())

        # Check pairs for compromise potential
        for i, (agent1, pos1) in enumerate(positions_list):
            for agent2, pos2 in positions_list[i + 1 :]:
                # Very simple check: same key terms but different emphasis
                pos1_words = set(pos1.lower().split())
                pos2_words = set(pos2.lower().split())

                overlap = pos1_words.intersection(pos2_words)
                if len(overlap) >= 3:  # Some common ground
                    # Potential compromise
                    compromise_position = (
                        f"Synthesized view combining {agent1}'s emphasis on "
                        f"[key aspect 1] with {agent2}'s focus on [key aspect 2]"
                    )

                    return Compromise(
                        iteration=iteration,
                        agents=[agent1, agent2],
                        original_positions={agent1: pos1, agent2: pos2},
                        compromise_position=compromise_position,
                        rationale="Detected common ground in positions",
                    )

        return None

    def build_consensus_items(
        self,
        agent_contributions: Dict[str, List[str]],
    ) -> tuple[List[str], List[str]]:
        """
        Identify consensus items (all agree) vs divergent items

        Args:
            agent_contributions: Agent -> list of key points

        Returns:
            (consensus_items, divergent_items)
        """
        # Flatten all contributions
        all_points = []
        point_to_agents: Dict[str, Set[str]] = {}

        for agent, points in agent_contributions.items():
            for point in points:
                # Normalize point
                normalized = point.lower().strip()
                all_points.append(normalized)

                if normalized not in point_to_agents:
                    point_to_agents[normalized] = set()
                point_to_agents[normalized].add(agent)

        # Consensus: mentioned by all or most agents
        num_agents = len(agent_contributions)
        consensus_threshold = num_agents * 0.7  # 70% agreement

        consensus = []
        divergent = []

        for point, agents in point_to_agents.items():
            if len(agents) >= consensus_threshold:
                consensus.append(point)
            elif len(agents) == 1:  # Only one agent
                divergent.append(point)

        return consensus, divergent

    def generate_collaboration_prompt(
        self,
        personality: Personality,
        question: str,
        strategy: CollaborationStrategy,
        shared_workspace: Dict[str, any],
        group_info: Optional[SubGroup] = None,
    ) -> str:
        """
        Generate system prompt for collaborative mode

        Args:
            personality: Agent personality
            question: Main question
            strategy: Collaboration strategy
            shared_workspace: Current shared information
            group_info: Sub-group information if applicable

        Returns:
            System prompt string
        """
        base = f"""Kamu adalah '{personality.name}'.

TRAITS: {personality.traits}
PERSPEKTIF: {personality.perspective}

MODE: COLLABORATIVE (bukan kompetitif!)
STRATEGI: {strategy.value}

ATURAN KOLABORASI:
1. FOKUS pada membangun solusi bersama, bukan menang argumen
2. Dengarkan dan acknowledge kontribusi agent lain
3. Cari common ground dan titik kesepakatan
4. Propose compromises bila ada perbedaan
5. Build on ide orang lain (gunakan "building on X's point...")
6. Bertanya untuk klarifikasi, bukan untuk menyerang
7. Goal: Solusi terintegrasi yang memanfaatkan semua perspektif

"""

        if group_info:
            base += f"""
SUB-GROUP INFO:
- Grup: {group_info.name}
- Focus area: {group_info.focus}
- Anggota: {', '.join([m.name for m in group_info.members])}
- Koordinator: {group_info.coordinator}

Kerja sama dalam grup untuk tackle {group_info.focus}.
"""

        if shared_workspace:
            base += f"""
SHARED WORKSPACE (akses semua):
"""
            for key, value in shared_workspace.items():
                base += f"- {key}: {value}\n"

        strategy_guidance = {
            CollaborationStrategy.CONSENSUS_BUILDING: (
                "Prioritas: Build consensus. Identify agreements, "
                "address disagreements dengan dialog konstruktif."
            ),
            CollaborationStrategy.PROBLEM_SOLVING: (
                "Prioritas: Solve problem together. Break down problem, "
                "contribute expertise, integrate solutions."
            ),
            CollaborationStrategy.SYNTHESIS: (
                "Prioritas: Synthesize viewpoints. Find complementary "
                "aspects, create holistic perspective."
            ),
            CollaborationStrategy.BRAINSTORMING: (
                "Prioritas: Generate ideas. Build on others, "
                "defer judgment, quantity over quality."
            ),
        }

        base += f"\nSTRATEGY GUIDANCE: {strategy_guidance[strategy]}\n"

        return base


def create_collaborative_debate_config(
    question: str,
    personalities: List[Personality],
    strategy: CollaborationStrategy = CollaborationStrategy.CONSENSUS_BUILDING,
    use_subgroups: bool = False,
    num_subgroups: int = 2,
) -> tuple[CollaborationEngine, Optional[List[SubGroup]]]:
    """
    Create configuration for collaborative debate

    Args:
        question: Main question/problem
        personalities: Participating agents
        strategy: Collaboration strategy
        use_subgroups: Whether to form sub-groups
        num_subgroups: Number of sub-groups if enabled

    Returns:
        (CollaborationEngine, Optional list of SubGroups)
    """
    engine = CollaborationEngine(strategy=strategy)

    subgroups = None
    if use_subgroups:
        subgroups = engine.form_subgroups(
            personalities,
            num_groups=num_subgroups,
            strategy="balanced",
        )
        engine.state.groups = subgroups

    return engine, subgroups
