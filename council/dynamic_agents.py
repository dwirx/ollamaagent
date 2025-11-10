from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langfuse.openai import OpenAI

from .types import Personality


class AgentCreationRequest(BaseModel):
    """Request to create a custom agent dynamically"""
    name: str
    domain: Optional[str] = None  # e.g., "medical ethics", "quantum physics"
    traits: Optional[str] = None  # User-defined traits
    perspective: Optional[str] = None  # User-defined perspective
    model: str = "qwen2.5:3b"
    reasoning_depth: int = Field(2, ge=1, le=3)
    truth_seeking: float = Field(0.8, ge=0.0, le=1.0)
    persistence: float = Field(0.6, ge=0.0, le=1.0)
    temperature: float = Field(0.7, ge=0.0, le=2.0)  # NEW: Per-agent temperature
    creativity: float = Field(0.5, ge=0.0, le=1.0)  # NEW: Creativity parameter


class DynamicAgentFactory:
    """
    Factory for creating agents dynamically:
    1. User-defined custom personalities
    2. Auto-generated from domain expertise
    3. Parameter tuning per agent
    """

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client

    def create_custom_agent(self, request: AgentCreationRequest) -> Personality:
        """
        Create a custom agent from user specifications

        Args:
            request: Agent creation parameters

        Returns:
            Personality object ready for debate
        """
        # If traits/perspective not provided, generate them
        if not request.traits or not request.perspective:
            if request.domain and self.client:
                # Auto-generate from domain
                generated = self._generate_from_domain(request.domain, request.name)
                traits = request.traits or generated["traits"]
                perspective = request.perspective or generated["perspective"]
            else:
                # Use defaults
                traits = request.traits or "Analytical, logical, evidence-based"
                perspective = request.perspective or "Evaluate arguments objectively with critical thinking"
        else:
            traits = request.traits
            perspective = request.perspective

        return Personality(
            name=request.name,
            model=request.model,
            traits=traits,
            perspective=perspective,
            reasoning_depth=request.reasoning_depth,
            truth_seeking=request.truth_seeking,
            persistence=request.persistence,
        )

    def _generate_from_domain(self, domain: str, name: str) -> Dict[str, str]:
        """
        Auto-generate personality traits and perspective from domain expertise

        Args:
            domain: Domain of expertise (e.g., "medical ethics")
            name: Agent name

        Returns:
            Dict with 'traits' and 'perspective' keys
        """
        if not self.client:
            return {
                "traits": f"Expert in {domain}, analytical, domain-focused",
                "perspective": f"Evaluate from {domain} perspective",
            }

        prompt = f"""Generate a debate agent personality for the domain: "{domain}"

Agent name: {name}

Provide:
1. Traits (3-5 key characteristics for this domain expert)
2. Perspective (their analytical framework and how they evaluate arguments)

Format your response as JSON:
{{
  "traits": "trait1, trait2, trait3, ...",
  "perspective": "description of their perspective..."
}}

Examples:
- Domain "medical ethics" → traits: "Hippocratic, patient-centered, risk-aware", perspective: "Evaluate through lens of patient welfare, medical evidence, and ethical principles"
- Domain "cybersecurity" → traits: "Paranoid productive, threat-modeling mindset, defense-in-depth", perspective: "Assess security implications, attack vectors, and mitigation strategies"

Generate for: {domain}
"""

        try:
            response = self.client.chat.completions.create(
                model="qwen2.5:3b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating debate agent personalities. Reply ONLY with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
            )

            import json
            result_text = response.choices[0].message.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)

            return {
                "traits": result.get("traits", f"Expert in {domain}"),
                "perspective": result.get("perspective", f"Analyze from {domain} viewpoint"),
            }

        except Exception as e:
            print(f"Error auto-generating personality: {e}")
            return {
                "traits": f"Expert in {domain}, analytical, thorough",
                "perspective": f"Evaluate arguments from {domain} perspective with rigor",
            }

    def create_domain_council(
        self,
        domain: str,
        num_agents: int = 5,
        include_critic: bool = True,
    ) -> List[Personality]:
        """
        Auto-create a full council specialized in a domain

        Args:
            domain: Domain (e.g., "medical ethics", "climate policy")
            num_agents: Number of agents to create
            include_critic: Whether to include a devil's advocate

        Returns:
            List of Personality objects
        """
        # Generate different perspectives within the domain
        perspectives = self._generate_domain_perspectives(domain, num_agents - (1 if include_critic else 0))

        agents = []
        for i, persp in enumerate(perspectives):
            request = AgentCreationRequest(
                name=f"{domain.title()} Expert {i+1}",
                domain=domain,
                traits=persp["traits"],
                perspective=persp["perspective"],
                reasoning_depth=2,
                truth_seeking=0.8,
            )
            agents.append(self.create_custom_agent(request))

        # Add critic if requested
        if include_critic:
            critic_request = AgentCreationRequest(
                name=f"{domain.title()} Skeptic",
                domain=domain,
                traits="Contrarian, critical, challenges assumptions",
                perspective=f"Critique {domain} arguments from skeptical viewpoint, expose weaknesses",
                reasoning_depth=3,
                truth_seeking=0.95,
            )
            agents.append(self.create_custom_agent(critic_request))

        return agents

    def _generate_domain_perspectives(self, domain: str, count: int) -> List[Dict[str, str]]:
        """Generate diverse perspectives within a domain"""
        if not self.client:
            # Fallback: generic perspectives
            return [
                {
                    "traits": f"{domain} specialist, analytical",
                    "perspective": f"Perspective {i+1} in {domain}",
                }
                for i in range(count)
            ]

        prompt = f"""Generate {count} diverse perspectives for experts in: "{domain}"

These should be DIFFERENT viewpoints within the same domain. Examples:
- Medical ethics: (1) Patient autonomy focus, (2) Public health utilitarian, (3) Professional ethics, (4) Cost-effectiveness
- Climate policy: (1) Economic transition, (2) Environmental justice, (3) Technological solutions, (4) Policy pragmatist

Provide {count} perspectives as JSON array:
[
  {{"traits": "...", "perspective": "..."}},
  ...
]
"""

        try:
            response = self.client.chat.completions.create(
                model="qwen2.5:3b",
                messages=[
                    {"role": "system", "content": "Generate diverse expert perspectives. Reply ONLY with JSON array."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
            )

            import json
            result_text = response.choices[0].message.content.strip()

            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            perspectives = json.loads(result_text)
            return perspectives[:count]

        except Exception as e:
            print(f"Error generating perspectives: {e}")
            return [
                {
                    "traits": f"{domain} expert, analytical, perspective {i+1}",
                    "perspective": f"Analyze {domain} from angle {i+1}",
                }
                for i in range(count)
            ]


# Pre-defined templates for quick access
DOMAIN_TEMPLATES = {
    "medical": {
        "name": "Medical Council",
        "domain": "medical ethics and healthcare",
        "description": "Experts in medical ethics, patient care, public health, and healthcare policy",
    },
    "legal": {
        "name": "Legal Council",
        "domain": "law and jurisprudence",
        "description": "Legal experts covering constitutional law, ethics, precedent, and justice",
    },
    "tech": {
        "name": "Technology Council",
        "domain": "technology and software engineering",
        "description": "Tech experts in software, security, scalability, and innovation",
    },
    "climate": {
        "name": "Climate Council",
        "domain": "climate science and environmental policy",
        "description": "Climate scientists, policy experts, economists, and activists",
    },
    "business": {
        "name": "Business Strategy Council",
        "domain": "business strategy and economics",
        "description": "Business strategists, economists, market analysts, and entrepreneurs",
    },
    "education": {
        "name": "Education Council",
        "domain": "education and pedagogy",
        "description": "Educators, policy makers, researchers, and student advocates",
    },
}


def get_domain_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get pre-defined domain template"""
    return DOMAIN_TEMPLATES.get(template_name.lower())


def list_domain_templates() -> List[str]:
    """List available domain templates"""
    return list(DOMAIN_TEMPLATES.keys())
