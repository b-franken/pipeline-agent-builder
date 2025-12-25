"""Structured output schemas for agents."""

from pydantic import BaseModel, Field


class CodeOutput(BaseModel):
    language: str = Field(description="Programming language")
    code: str = Field(description="Generated code without markdown fences")
    explanation: str = Field(default="", description="Brief explanation")
    dependencies: list[str] = Field(default_factory=list, description="Required packages")


class CodeReviewOutput(BaseModel):
    issues: list[str] = Field(default_factory=list, description="Issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvements")
    security_concerns: list[str] = Field(default_factory=list, description="Security issues")
    overall_quality: str = Field(description="good, needs_improvement, or critical_issues")


class ResearchOutput(BaseModel):
    summary: str = Field(description="Concise summary of findings")
    sources: list[str] = Field(default_factory=list, description="URLs or references")
    key_facts: list[str] = Field(default_factory=list, description="Key facts discovered")
    confidence: str = Field(default="medium", description="high, medium, or low")


class TaskPlanOutput(BaseModel):
    steps: list[str] = Field(description="Ordered steps to complete the task")
    agent_assignments: dict[str, str] = Field(default_factory=dict, description="Step to agent mapping")
    estimated_complexity: str = Field(default="medium", description="simple, medium, or complex")
