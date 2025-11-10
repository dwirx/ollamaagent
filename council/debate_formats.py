from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from .types import Personality


class DebateFormat(Enum):
    """Supported debate formats"""
    FREEFORM = "freeform"  # Original unstructured
    OXFORD = "oxford"  # Oxford-style: proposition vs opposition
    SOCRATIC = "socratic"  # Socratic questioning method
    DEVILS_ADVOCATE = "devils_advocate"  # One agent challenges all
    PARLIAMENTARY = "parliamentary"  # Parliamentary rules with motions
    FISHBOWL = "fishbowl"  # Inner/outer circle discussion


@dataclass
class OxfordDebateConfig:
    """Oxford-style debate configuration"""
    motion: str  # The proposition being debated
    proposition_team: List[Personality]
    opposition_team: List[Personality]
    opening_statement_time: int = 1  # Iterations for opening
    rebuttal_rounds: int = 2
    closing_statement_time: int = 1


@dataclass
class SocraticConfig:
    """Socratic questioning configuration"""
    questioner: Personality  # The Socrates figure
    respondents: List[Personality]
    question_depth: int = 3  # How many follow-up questions per topic
    focus_on_definitions: bool = True
    expose_assumptions: bool = True


@dataclass
class DevilsAdvocateConfig:
    """Devil's Advocate configuration"""
    devil: Personality  # The challenger
    proponents: List[Personality]  # Those being challenged
    challenge_every_claim: bool = True
    require_evidence: bool = True


@dataclass
class ParliamentaryConfig:
    """Parliamentary debate configuration"""
    motion: str
    government_team: List[Personality]  # Propose motion
    opposition_team: List[Personality]  # Oppose motion
    speaker_order: List[str]  # Ordered list of speaker names
    points_of_order_allowed: bool = True
    points_of_information_allowed: bool = True


class DebateFormatEngine:
    """Engine for managing different debate formats"""

    def __init__(self, format_type: DebateFormat):
        self.format = format_type
        self.config: Optional[Any] = None

    def generate_format_prompt(
        self,
        personality: Personality,
        role: str,
        phase: str,
        format_config: Any,
    ) -> str:
        """
        Generate format-specific system prompt

        Args:
            personality: Agent personality
            role: Role in this format (e.g., "proposition", "questioner")
            phase: Current phase
            format_config: Format-specific configuration

        Returns:
            System prompt
        """
        base = f"""Kamu adalah '{personality.name}'.

TRAITS: {personality.traits}
PERSPEKTIF: {personality.perspective}

"""

        if self.format == DebateFormat.OXFORD:
            return self._oxford_prompt(personality, role, phase, format_config)

        elif self.format == DebateFormat.SOCRATIC:
            return self._socratic_prompt(personality, role, phase, format_config)

        elif self.format == DebateFormat.DEVILS_ADVOCATE:
            return self._devils_advocate_prompt(personality, role, phase, format_config)

        elif self.format == DebateFormat.PARLIAMENTARY:
            return self._parliamentary_prompt(personality, role, phase, format_config)

        return base

    def _oxford_prompt(
        self,
        personality: Personality,
        role: str,
        phase: str,
        config: OxfordDebateConfig,
    ) -> str:
        """Oxford-style debate prompt"""
        base = f"""
FORMAT: OXFORD DEBATE
MOTION: "{config.motion}"
YOUR ROLE: {role.upper()}
PHASE: {phase}

OXFORD RULES:
1. Proposition team SUPPORTS the motion
2. Opposition team OPPOSES the motion
3. Structure: Opening → Rebuttals → Closing
4. Stay in character of your side
5. Address judge and audience, not opponents directly
6. Use formal, persuasive language

"""

        if role == "proposition":
            base += f"""
TUGAS ANDA:
- Defend the motion dengan argumen kuat
- Provide evidence dan examples
- Anticipate opposition arguments
- Build case yang persuasif untuk audience
"""
        elif role == "opposition":
            base += f"""
TUGAS ANDA:
- Challenge the motion dengan counter-arguments
- Expose weaknesses dalam proposition case
- Provide alternative perspectives
- Persuade audience motion should NOT pass
"""

        if phase == "opening":
            base += "\n➤ OPENING STATEMENT: Lay out your main arguments (3-5 poin kunci)"
        elif phase == "rebuttal":
            base += "\n➤ REBUTTAL: Address opponent arguments AND reinforce your case"
        elif phase == "closing":
            base += "\n➤ CLOSING: Summarize why your side should win, address key clashes"

        return base

    def _socratic_prompt(
        self,
        personality: Personality,
        role: str,
        phase: str,
        config: SocraticConfig,
    ) -> str:
        """Socratic questioning prompt"""
        base = f"""
FORMAT: SOCRATIC DIALOGUE
YOUR ROLE: {role.upper()}
PHASE: {phase}

SOCRATIC METHOD:
- Questioner asks probing questions to expose assumptions
- Respondents answer and refine their thinking
- Goal: Reach deeper understanding through dialectic

"""

        if role == "questioner":
            base += f"""
TUGAS ANDA (SOCRATES):
1. Ask clarifying questions: "What do you mean by X?"
2. Probe assumptions: "What are you assuming when you say Y?"
3. Question evidence: "How do you know that?"
4. Explore implications: "If that's true, what follows?"
5. Challenge viewpoints: "What alternative perspectives exist?"

ATURAN:
- HANYA bertanya, JANGAN argue
- Questions harus genuine, bukan rhetorical attacks
- Guide respondent to examine their own beliefs
- {"Focus on definitions" if config.focus_on_definitions else ""}
- {"Expose hidden assumptions" if config.expose_assumptions else ""}

Format: "Question: [your question]"
"""
        elif role == "respondent":
            base += f"""
TUGAS ANDA (RESPONDENT):
- Answer questions thoughtfully dan honestly
- Examine your own assumptions when questioned
- Refine your position based on dialog
- Admit uncertainty bila tidak yakin

ATURAN:
- Jangan defensive
- Think through implications dari answer Anda
- Collaborate dalam pencarian kebenaran
"""

        return base

    def _devils_advocate_prompt(
        self,
        personality: Personality,
        role: str,
        phase: str,
        config: DevilsAdvocateConfig,
    ) -> str:
        """Devil's Advocate prompt"""
        base = f"""
FORMAT: DEVIL'S ADVOCATE
YOUR ROLE: {role.upper()}

"""

        if role == "devil":
            base += f"""
TUGAS ANDA (DEVIL'S ADVOCATE):
Anda adalah CHALLENGER yang bertugas:

1. ✓ Challenge EVERY claim yang dibuat
2. ✓ Demand evidence untuk assertions
3. ✓ Expose weak logic dan assumptions
4. ✓ Play contrarian to stress-test ideas
5. ✓ Ask "What if...?" scenarios
6. ✓ Point out edge cases dan exceptions

ATURAN:
- Challenge constructively, bukan destructively
- Goal: Strengthen ideas melalui scrutiny
- Use "Devil's Advocate" phrases:
  * "Let me challenge that..."
  * "What evidence supports...?"
  * "Consider this counter-example..."
  * "That assumes X, but what if...?"

{"REQUIRE EVIDENCE: Selalu minta bukti untuk claims" if config.require_evidence else ""}
"""
        elif role == "proponent":
            base += f"""
TUGAS ANDA (PROPONENT):
- Present your ideas dengan clear logic
- Provide evidence saat di-challenge
- Defend positions dengan reasoning kuat
- Acknowledge valid criticisms
- Refine arguments based on feedback

ATURAN:
- Expect to be challenged on everything
- This is HELPFUL - makes your ideas stronger
- Don't take challenges personally
"""

        return base

    def _parliamentary_prompt(
        self,
        personality: Personality,
        role: str,
        phase: str,
        config: ParliamentaryConfig,
    ) -> str:
        """Parliamentary debate prompt"""
        base = f"""
FORMAT: PARLIAMENTARY DEBATE
MOTION: "{config.motion}"
YOUR ROLE: {role.upper()}
PHASE: {phase}

PARLIAMENTARY RULES:
1. Government defends motion, Opposition opposes
2. Formal speaker order must be followed
3. Address the "Speaker" (moderator)
4. Points of Order: Challenge rule violations
5. Points of Information: Brief interjections (15 sec)
6. No interrupting except for procedural points

FORMAL LANGUAGE:
- "Honorable Speaker..."
- "The Honorable member from..."
- "I rise to make a Point of Order..."
- "Will the speaker yield for a Point of Information?"

"""

        if role == "government":
            base += f"""
GOVERNMENT TEAM:
- Propose and defend the motion
- Burden of proof is on YOU
- Define terms fairly but favorably
- Build constructive case

STRATEGY:
- Opening: Define motion, lay out case
- Middle: Reinforce points, rebut opposition
- Closing: Summarize why motion should pass
"""
        elif role == "opposition":
            base += f"""
OPPOSITION TEAM:
- Challenge and oppose the motion
- Rebut government case
- Provide alternative framework
- No burden to propose solution (unless motion requires)

STRATEGY:
- Opening: Challenge definitions, set up opposition case
- Middle: Tear down government arguments
- Closing: Summarize why motion should fail
"""

        if config.points_of_order_allowed:
            base += "\n✓ Points of Order ALLOWED: Flag rule violations"

        if config.points_of_information_allowed:
            base += "\n✓ Points of Information ALLOWED: Brief interjections to opponent"

        return base


def create_oxford_debate(
    motion: str,
    all_agents: List[Personality],
    proposition_count: int = 3,
) -> OxfordDebateConfig:
    """
    Create Oxford-style debate configuration

    Args:
        motion: The motion to debate
        all_agents: All available agents
        proposition_count: Number of agents for proposition (rest go to opposition)

    Returns:
        OxfordDebateConfig
    """
    prop_team = all_agents[:proposition_count]
    opp_team = all_agents[proposition_count:]

    return OxfordDebateConfig(
        motion=motion,
        proposition_team=prop_team,
        opposition_team=opp_team,
        opening_statement_time=1,
        rebuttal_rounds=2,
        closing_statement_time=1,
    )


def create_socratic_dialog(
    questioner: Personality,
    respondents: List[Personality],
    depth: int = 3,
) -> SocraticConfig:
    """Create Socratic questioning configuration"""
    return SocraticConfig(
        questioner=questioner,
        respondents=respondents,
        question_depth=depth,
        focus_on_definitions=True,
        expose_assumptions=True,
    )


def create_devils_advocate_debate(
    devil: Personality,
    proponents: List[Personality],
) -> DevilsAdvocateConfig:
    """Create Devil's Advocate configuration"""
    return DevilsAdvocateConfig(
        devil=devil,
        proponents=proponents,
        challenge_every_claim=True,
        require_evidence=True,
    )


def create_parliamentary_debate(
    motion: str,
    all_agents: List[Personality],
    government_count: int = 3,
) -> ParliamentaryConfig:
    """Create Parliamentary debate configuration"""
    gov_team = all_agents[:government_count]
    opp_team = all_agents[government_count:]

    # Alternating speaker order
    speaker_order = []
    max_len = max(len(gov_team), len(opp_team))
    for i in range(max_len):
        if i < len(gov_team):
            speaker_order.append(gov_team[i].name)
        if i < len(opp_team):
            speaker_order.append(opp_team[i].name)

    return ParliamentaryConfig(
        motion=motion,
        government_team=gov_team,
        opposition_team=opp_team,
        speaker_order=speaker_order,
        points_of_order_allowed=True,
        points_of_information_allowed=True,
    )
