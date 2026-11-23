"""Bounded, provenance-preserving repair for imperfect local LLM JSON.

This model adapter never invents evidence/claim IDs. An unsupported project
still fails closed; optional unsafe skill merges can be ignored, and individual
skills without known evidence may be omitted only as a last resort.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from github2cv_ml.cv_writer import ProposedOverview
from github2cv_ml.profile_builder import (
    ProposedProject,
    ProposedSkill,
    RepositoryProposal,
    SkillReconciliation,
)

MAX_ATTEMPTS = 3


class RecoveringModel:
    """Wrap a ChatModel for profile and CV repair, recording audit events.

    Explorer, extractor and all non-target calls pass through unchanged.
    Events contain no raw prompts, model output or source excerpts.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def _response_data(answer: Any) -> Any:
        try:
            if not isinstance(answer, dict) or not isinstance(answer.get("content"), str):
                return None
            return json.loads(answer["content"])
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _format_error(exc: Exception) -> str:
        if isinstance(exc, ValidationError):
            return "Invalid schema: " + "; ".join(
                f"{'.'.join(map(str, issue['loc']))}: {issue['type']}"
                for issue in exc.errors()[:5]
            )
        return (str(exc) or type(exc).__name__)[:250]

    def _complete_with_feedback(
        self, messages: list[dict[str, Any]], feedback: str | None, *,
        stage: str, tools: Any, json_mode: bool,
    ) -> Any:
        if feedback is None:
            prompt = messages
        else:
            if stage == "repository_profile":
                instruction = (
                    "SCHEMA AND EVIDENCE REPAIR: Previous response was rejected: "
                    f"{feedback}. Return the full JSON object, with skills "
                    "(each with name and NONEMPTY evidence_ids) and project "
                    "(paragraph and NONEMPTY evidence_ids). Copy ONLY exact ids "
                    "from the original observations. Remove unsupported skills. "
                    "Never invent or guess evidence IDs."
                )
            else:
                instruction = (
                    "CV SCHEMA AND CLAIM REPAIR: Previous response was rejected: "
                    f"{feedback}. Return full JSON summary and optional headline. "
                    "Use only existing claim_ids from the input; no made-up "
                    "technologies, seniority or achievements. Do NOT put claim "
                    "IDs or evidence IDs into visible text, only in claim_ids."
                )
            prompt = [
                {"role": "system", "content": messages[0]["content"] + "\n" + instruction},
                *messages[1:],
            ]
        return self.model.complete(prompt, tools=tools, json_mode=json_mode)

    @staticmethod
    def _supported_proposal(data: Any, allowed: set[str]) -> RepositoryProposal:
        proposal = RepositoryProposal.model_validate(data)
        references = [
            identifier for skill in proposal.skills for identifier in skill.evidence_ids
        ]
        references.extend(proposal.project.evidence_ids)
        unknown = set(references) - allowed
        if unknown:
            raise ValueError("Unknown evidence_ids: " + ", ".join(sorted(unknown)[:5]))
        # Check inline evidence markers before the model output reaches the
        # profile builder: bad visible citations must trigger a proper retry.
        visible = set(re.findall(
            r"(?<![\w-])ev-[0-9a-f]{16}(?![\w-])",
            proposal.project.paragraph,
        ))
        if visible - set(proposal.project.evidence_ids):
            raise ValueError(
                "Unreferenced inline evidence_ids: "
                + ", ".join(sorted(visible - set(proposal.project.evidence_ids))[:5])
            )
        return proposal

    @staticmethod
    def _drop_unsupported_skills(
        data: Any, allowed: set[str]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Omit bad skills only when the *entire* project has real citations."""
        if not isinstance(data, dict) or not isinstance(data.get("skills"), list):
            return None, []
        try:
            project = ProposedProject.model_validate(data.get("project"))
        except (TypeError, ValueError, ValidationError):
            return None, []
        if not set(project.evidence_ids).issubset(allowed):
            return None, []
        kept: list[dict[str, Any]] = []
        dropped: list[str] = []
        for raw in data["skills"]:
            try:
                skill = ProposedSkill.model_validate(raw)
            except (TypeError, ValueError, ValidationError):
                dropped.append(
                    str(raw.get("name", "<invalid>"))[:64]
                    if isinstance(raw, dict) else "<invalid>"
                )
                continue
            if set(skill.evidence_ids).issubset(allowed):
                kept.append(skill.model_dump())
            else:
                dropped.append(skill.name)
        if not dropped or len(kept) > 20:
            return None, []
        safe = {"skills": kept, "project": project.model_dump()}
        RecoveringModel._supported_proposal(safe, allowed)
        return safe, dropped

    def _repo_profile(
        self, messages: list[dict[str, Any]], payload: dict[str, Any], *,
        tools: Any, json_mode: bool,
    ) -> dict[str, Any]:
        allowed = {
            entry["id"] for entry in payload["observations"]
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }
        if not allowed:
            return self.model.complete(messages, tools=tools, json_mode=json_mode)
        repository = payload["repository"]
        feedback: str | None = None
        last_data: Any = None
        last_answer: Any = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            answer = self._complete_with_feedback(
                messages, feedback, stage="repository_profile",
                tools=tools, json_mode=json_mode,
            )
            last_answer = answer
            last_data = self._response_data(answer)
            try:
                self._supported_proposal(last_data, allowed)
                return answer
            except (TypeError, ValueError, ValidationError) as exc:
                feedback = self._format_error(exc)
                self.events.append({
                    "stage": "repository_profile", "repository": repository,
                    "attempt": attempt, "kind": "retry", "reason": feedback,
                })
        safe, dropped = self._drop_unsupported_skills(last_data, allowed)
        if safe is not None:
            self.events.append({
                "stage": "repository_profile", "repository": repository,
                "kind": "omitted_unsupported_skills", "count": len(dropped),
                "skill_names": dropped,
            })
            return {
                **(last_answer if isinstance(last_answer, dict) else {}),
                "content": json.dumps(safe, ensure_ascii=False),
            }
        # Never reconstruct unsupported project evidence from unrelated skills.
        return last_answer

    @staticmethod
    def _safe_merges(
        data: Any, names: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], int]:
        names_by_key = {
            entry["name"].strip().casefold(): entry["name"]
            for entry in names
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str) and entry["name"].strip()
        }
        if not isinstance(data, dict) or not isinstance(data.get("merges"), list):
            return {"merges": []}, 1
        if len(data["merges"]) > 40:
            return {"merges": []}, len(data["merges"])
        used: set[str] = set()
        merged: list[dict[str, Any]] = []
        skipped = 0
        for group in data["merges"]:
            if not isinstance(group, dict):
                skipped += 1
                continue
            canonical = group.get("canonical")
            aliases = group.get("aliases")
            reason = group.get("reason")
            if (
                not isinstance(canonical, str)
                or not isinstance(aliases, list) or not 1 <= len(aliases) <= 32
                or not isinstance(reason, str) or not 10 <= len(reason.strip()) <= 300
            ):
                skipped += 1
                continue
            key = canonical.strip().casefold()
            if key not in names_by_key or key in used:
                skipped += 1
                continue
            valid: list[str] = []
            seen = {key}
            invalid = False
            for alias in aliases:
                if not isinstance(alias, str) or alias.strip().casefold() not in names_by_key:
                    invalid = True
                    break
                alias_key = alias.strip().casefold()
                if alias_key in seen:
                    continue  # self-alias and duplicate alias are harmless no-ops
                seen.add(alias_key)
                valid.append(names_by_key[alias_key])
            if invalid or not valid or used & seen:
                skipped += 1
                continue
            group_safe = {
                "canonical": names_by_key[key], "aliases": valid, "reason": reason.strip()
            }
            try:
                SkillReconciliation.model_validate({"merges": [group_safe]})
            except (TypeError, ValueError, ValidationError):
                skipped += 1
                continue
            merged.append(group_safe)
            used.update(seen)
        return {"merges": merged}, skipped

    def _reconcile(
        self, messages: list[dict[str, Any]], payload: dict[str, Any], *,
        tools: Any, json_mode: bool,
    ) -> dict[str, Any]:
        answer = self.model.complete(messages, tools=tools, json_mode=json_mode)
        safe, skipped = self._safe_merges(
            self._response_data(answer), payload.get("skills", [])
        )
        if skipped:
            self.events.append({
                "stage": "skill_reconciliation", "kind": "ignored_invalid_merges",
                "skipped": skipped, "kept": len(safe["merges"]),
            })
        return {
            **(answer if isinstance(answer, dict) else {}),
            "content": json.dumps(safe, ensure_ascii=False),
        }

    @staticmethod
    def _validated_overview(data: Any, available: set[str]) -> ProposedOverview:
        overview = ProposedOverview.model_validate(data)
        for label, item in (("summary", overview.summary), ("headline", overview.headline)):
            if item is None:
                continue
            # Validate visible citation tokens too: the core Writer later
            # strips *known* claim markers into claim_ids, never unknown ones.
            embedded = set(re.findall(
                r"(?<![\w-])claim-[a-z]+-[0-9a-f]{16}(?![\w-])", item.text
            ))
            if embedded - available:
                raise ValueError(f"Unknown inline claim ID in {label}")
            if re.search(r"(?<![\w-])ev-[0-9a-f]{16}(?![\w-])", item.text):
                raise ValueError(f"Inline evidence ID in {label}")
            ids = item.claim_ids
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate claim_ids in {label}")
            unknown = set(ids) - available
            if unknown:
                raise ValueError(
                    f"Unknown claim_ids in {label}: {', '.join(sorted(unknown)[:5])}"
                )
        return overview

    @staticmethod
    def _conservative_overview(claims: list[dict[str, Any]]) -> dict[str, Any]:
        projects = [claim for claim in claims if claim.get("kind") == "project"]
        skills = [claim for claim in claims if claim.get("kind") == "skill"]
        if projects:
            selected = projects[:3]
            text = (
                "В портфолио представлены проекты, сведения о которых получены "
                "из анализа доступных репозиториев."
            )
        elif skills:
            selected = skills[:1]
            text = (
                "В изученных репозиториях представлены технические сведения "
                "о применяемых инструментах и технологиях."
            )
        else:
            selected = claims[:1]
            text = "В портфолио представлены технические сведения из репозиториев."
        return {
            "summary": {"text": text, "claim_ids": [claim["id"] for claim in selected]},
            "headline": None,
        }

    def _cv_overview(
        self, messages: list[dict[str, Any]], payload: dict[str, Any], *,
        tools: Any, json_mode: bool,
    ) -> dict[str, Any]:
        claims = [
            item for item in payload.get("claims", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
            and isinstance(item.get("kind"), str)
        ]
        available = {claim["id"] for claim in claims}
        if not available:
            return self.model.complete(messages, tools=tools, json_mode=json_mode)
        feedback: str | None = None
        answer: Any = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            answer = self._complete_with_feedback(
                messages, feedback, stage="cv_overview",
                tools=tools, json_mode=json_mode,
            )
            try:
                self._validated_overview(self._response_data(answer), available)
                return answer
            except (TypeError, ValueError, ValidationError) as exc:
                feedback = self._format_error(exc)
                self.events.append({
                    "stage": "cv_overview", "attempt": attempt,
                    "kind": "retry", "reason": feedback,
                })
        fallback = self._conservative_overview(claims)
        self._validated_overview(fallback, available)
        self.events.append({
            "stage": "cv_overview", "kind": "conservative_fallback",
            "claim_ids": fallback["summary"]["claim_ids"],
        })
        return {"content": json.dumps(fallback, ensure_ascii=False)}

    def complete(
        self, messages: list[dict[str, Any]], *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        if not json_mode or tools is not None or not messages:
            return self.model.complete(messages, tools=tools, json_mode=json_mode)
        try:
            payload = json.loads(messages[-1]["content"])
        except (ValueError, TypeError, KeyError):
            return self.model.complete(messages, tools=tools, json_mode=json_mode)
        if not isinstance(payload, dict):
            return self.model.complete(messages, tools=tools, json_mode=json_mode)
        if (
            isinstance(payload.get("repository"), str)
            and isinstance(payload.get("observations"), list)
        ):
            return self._repo_profile(messages, payload, tools=tools, json_mode=json_mode)
        if payload.get("task") == "skill_reconciliation":
            return self._reconcile(messages, payload, tools=tools, json_mode=json_mode)
        if payload.get("task") == "cv_overview":
            return self._cv_overview(messages, payload, tools=tools, json_mode=json_mode)
        return self.model.complete(messages, tools=tools, json_mode=json_mode)
